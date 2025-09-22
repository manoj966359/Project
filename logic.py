import os
import mysql.connector
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

client = OpenAI(api_key=api_key)

@st.cache_data(ttl=600)
def get_databases():
    conn = mysql.connector.connect(host='localhost', user='root', password='root')
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES")
    dbs = [db[0] for db in cursor.fetchall()]
    cursor.close()
    conn.close()
    return dbs

@st.cache_data(ttl=600)
def get_tables(database_name):
    conn = mysql.connector.connect(host='localhost', user='root', password='root', database=database_name)
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]
    cursor.close()
    conn.close()
    return tables

@st.cache_data(ttl=600)
def get_schema_details(database, tables):
    conn = mysql.connector.connect(host='localhost', user='root', password='root', database=database)
    cursor = conn.cursor()
    schema_str = ""
    schema_dict = {}
    for table in tables:
        cursor.execute(f"SHOW COLUMNS FROM `{table}`")
        columns = cursor.fetchall()
        col_list = [f"{col[0]}({col[1]})" for col in columns]
        schema_dict[table] = col_list
        schema_str += f"Table {table} columns: {', '.join(col_list)}\n"
    format_tables = ",".join([f"'{t}'" for t in tables]) if tables else "''"
    query = f"""
        SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME IN ({format_tables})
        AND REFERENCED_TABLE_NAME IS NOT NULL;
    """
    cursor.execute(query, (database,))
    relationships = cursor.fetchall()
    if relationships:
        schema_str += "Relationships:\n"
        for rel in relationships:
            schema_str += f"{rel[0]}.{rel[1]} -> {rel[2]}.{rel[3]}\n"
    cursor.close()
    conn.close()
    return schema_str, schema_dict

@st.cache_data(ttl=3600)
def get_schema_embeddings(schema_dict):
    texts = []
    keys = []
    for table, cols in schema_dict.items():
        for col in cols:
            key = f"{table}.{col}"
            keys.append(key)
            texts.append(key)
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embeddings = [np.array(data.embedding) for data in response.data]
    emb_dict = dict(zip(keys, embeddings))
    return emb_dict

def semantic_match(user_query, embeddings_dict, top_n=3):
    query_emb = client.embeddings.create(model="text-embedding-3-small", input=user_query).data[0].embedding
    query_vec = np.array(query_emb)
    candidates = []
    for key, emb in embeddings_dict.items():
        sim = cosine_similarity([query_vec], [emb])[0][0]
        candidates.append((key, sim))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]

def extract_valid_sql(sql_text):
    sql_text = sql_text.strip()
    for prefix in ["sql\n", "sql:", "SQL\n", "SQL:"]:
        if sql_text.lower().startswith(prefix.lower()):
            sql_text = sql_text[len(prefix):].lstrip()
    lines = sql_text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(("select", "insert", "update", "delete", "with")):
            return '\n'.join(lines[i:]).strip()
    return sql_text

def prompt_to_sql_openai(prompt, tables, schema_info, relevant_matches):
    system_prompt = f"""
You are an expert MySQL query generator.
The database has the following schema:
{schema_info}
Relevant tables/columns for this query: {', '.join(relevant_matches)}
If dates are stored as text in DD-MM-YYYY format, convert them with STR_TO_DATE. Example:
  DATE_FORMAT(STR_TO_DATE(Disbursement date, '%d-%m-%Y'), '%b-%y') AS MonthYear
When filtering or grouping by month or date, always:
- Use DATE_FORMAT(STR_TO_DATE(...)) for consistent grouping
- Include all non-aggregated columns in the GROUP BY clause
- Never use non-grouped columns in ORDER BY unless aggregated
#### Aggregation & Grouping
- Include ALL non-aggregated columns in GROUP BY clause
- Wrap aggregated columns with COALESCE
- Use HAVING for filtering aggregated values
#### SQL Query Rules:
- Use proper table aliases
- Enclose identifiers with spaces or special characters in backticks
- Support date conversion formats carefully
- Validate table and column names referenced
### OUTPUT FORMAT
- Return ONLY the SQL query without markdown formatting
- Include brief comments for complex logic
- Ensure query is executable and syntactically correct
- Use proper indentation for readability
Return only valid MySQL SQL SELECT query.
User request: {prompt}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert SQL query generator."},
            {"role": "user", "content": system_prompt}
        ],
        temperature=0,
    )
    sql_query = extract_valid_sql(response.choices[0].message.content)
    if not sql_query.lower().startswith(("select", "insert", "update", "delete", "with")):
        raise ValueError("The generated content does not appear to be a valid SQL query.")
    return sql_query

def run_query(database, query):
    conn = mysql.connector.connect(host='localhost', user='root', password='root', database=database)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
    except Exception as e:
        cursor.close()
        conn.close()
        raise e
    cursor.close()
    conn.close()
    return columns, rows
