import streamlit as st
import mysql.connector
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import io

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
Relevant tables/columns for this query: {', '.join(relevant_matches)}
If dates are stored as text in DD-MM-YYYY format, convert them with STR_TO_DATE. Example:
  DATE_FORMAT(STR_TO_DATE(Disbursement date, '%d-%m-%Y'), '%b-%y') AS MonthYear
When filtering or grouping by month or date, always:
- Use DATE_FORMAT(STR_TO_DATE(...)) for consistent grouping
- Include all non-aggregated columns in the GROUP BY clause
- Do not use non-grouped columns in ORDER BY without aggregation
#### Aggregation & Grouping
- Include ALL non-aggregated columns in GROUP BY clause
- Never use non-grouped columns in ORDER BY unless they are aggregated
- Wrap all aggregated columns with COALESCE to handle NULLs: `SUM(COALESCE(...))`
- Use HAVING clause for filtering aggregated results, not WHERE
...
### OUTPUT FORMAT
- Ensure group by compliance rules are strictly followed.
### CORE SQL GENERATION RULES
#### Identifier & Alias Management
- Always alias duplicate column names using unique descriptive names with `AS`
  Example: SELECT t1.ApplicationId AS App_ID_Primary, t2.ApplicationId AS App_ID_Secondary
- Use meaningful table aliases (avoid generic aliases like t1, t2, s1, sd1)
- Enclose all identifiers with spaces, special characters, or reserved words in backticks: `column name`
- Never use square brackets [ ] or positional references like [2]
- Use full table names when aliases are ambiguous
#### Date & Time Handling
- For dates in 'DD-MM-YYYY' format:
  `DATE_FORMAT(STR_TO_DATE(`date_column`, '%d-%m-%Y'), '%Y-%m-%d')`
- For dates in 'DD-MM-YYYY  HH:MM:SS' format:
  `DATE_FORMAT(STR_TO_DATE(`datetime_column`, '%d-%m-%Y  %H:%i:%s'), '%Y-%m-%d %H:%i:%s')`
- Always use STR_TO_DATE for text date conversion before filtering or grouping
- Use DATE_FORMAT for consistent date grouping and display
-If dates are stored as text in 'DD-MM-YYYY  HH:MM:SS' format, convert them with STR_TO_DATE using format '%d-%m-%Y  %H:%i:%s'. Example:
  DATE_FORMAT(STR_TO_DATE(`Disbursement date`, '%d-%m-%Y  %H:%i:%s'), '%b-%y') AS MonthYear
#### Aggregation & Grouping
- Wrap all aggregated columns with COALESCE to handle NULLs: `SUM(COALESCE(`amount`, 0))`
- Include ALL non-aggregated columns in GROUP BY clause
- Never use non-grouped columns in ORDER BY unless they are aggregated
- Use HAVING clause for filtering aggregated results, not WHERE
#### JOIN Operations
- Explicitly specify JOIN types (INNER, LEFT, RIGHT, FULL OUTER)
- Always include proper ON conditions for joins
- Use table aliases consistently throughout the query
- Consider performance implications of join order
#### String & Text Operations
- Use proper MySQL string functions: CONCAT(), SUBSTRING(), TRIM(), UPPER(), LOWER()
- Use LIKE with wildcards % for pattern matching
- Use REGEXP for complex pattern matching when needed
- Handle case sensitivity appropriately
#### NULL Handling
- Use IS NULL / IS NOT NULL for null comparisons
- Use COALESCE() or IFNULL() for default values
- Consider NULL behavior in aggregations and comparisons
#### Query Optimization
- Use appropriate indexes (mention in comments when relevant)
- Limit result sets when possible using LIMIT clause
- Use EXISTS instead of IN for subqueries when appropriate
- Avoid SELECT * unless specifically requested
#### Security & Validation
- Never generate queries that could lead to data modification without explicit request
- Avoid dynamic SQL construction that could lead to injection
- Validate that all referenced tables and columns exist in the schema
- Use parameterized approaches when dealing with user input values
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
def generate_pivot_table(df, index_cols, column_cols, agg_dict, show_pct):
    if not agg_dict:
        return pd.DataFrame({"Error": ["Select at least one value and aggregation function"]})

    pivot = pd.pivot_table(
        df,
        index=index_cols if index_cols else None,
        columns=column_cols if column_cols else None,
        values=list(agg_dict.keys()),
        aggfunc=agg_dict,
        fill_value=0,
    )
    pivot = pivot.reset_index()

    value_cols = pivot.select_dtypes(include="number").columns
    percent_df = pivot.copy()

    if show_pct == "Row %":
        percent_df[value_cols] = percent_df[value_cols].div(percent_df[value_cols].sum(axis=1), axis=0).fillna(0) * 100
    elif show_pct == "Column %":
        percent_df[value_cols] = percent_df[value_cols].div(percent_df[value_cols].sum(axis=0), axis=1).fillna(0) * 100
    elif show_pct == "Overall %":
        total = percent_df[value_cols].values.sum()
        percent_df[value_cols] = percent_df[value_cols].div(total).fillna(0) * 100
    else:
        percent_df[value_cols] = 0

    for col in value_cols:
        pivot[f"{col} (%)"] = percent_df[col].round(2)

    pivot.columns = [
        " ".join([str(c) for c in col if c and str(c) != ""]) if isinstance(col, tuple) else str(col)
        for col in pivot.columns
    ]
    return pivot



# Page configuration
st.set_page_config(
    page_title="Metatalk - Text to SQL",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    ..main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: white;                                           /* White text */
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    background: linear-gradient(90deg, #2c3e50, #34495e);  /* Dark background */
    border-radius: 10px;
    border: 2px solid #34495e;                             /* Dark border */
}

    .step-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .step-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .pivot-container {
        background-color: #fff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    .stButton > button {
    background: linear-gradient(90deg, #2c3e50, #34495e) !important;  /* Dark navy gradient */
    color: white !important;                                          /* White text */
    border: 2px solid #2c3e50 !important;                           /* Dark border */
    border-radius: 12px !important;                                 /* Rounded corners */
    padding: 0.7rem 2.5rem !important;                             /* More padding */
    font-weight: 700 !important;                                   /* Bold text */
    font-size: 16px !important;                                    /* Larger font */
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;           /* Text shadow */
    letter-spacing: 0.5px !important;                             /* Letter spacing */
    text-transform: uppercase !important;                          /* Uppercase text */
    box-shadow: 0 4px 8px rgba(44, 62, 80, 0.3) !important;      /* Button shadow */
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1a252f, #2c3e50) !important;  /* Darker on hover */
    color: #ffffff !important;                                         /* Keep white text */
    border-color: #1a252f !important;                                /* Darker border */
    box-shadow: 0 6px 15px rgba(26, 37, 47, 0.5) !important;        /* Stronger shadow */
    transform: translateY(-2px) !important;                           /* Slight lift effect */
}

    }
</style>    
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📊 Metatalk - Text to SQL with Advanced Pivot Tables</h1>', unsafe_allow_html=True)

databases = get_databases()

# Initialize session state
for key, default in [
    ("selected_db", None),
    ("selected_tables", []),
    ("user_prompt", ""),
    ("query_result_df", pd.DataFrame()),
    ("sql_query", ""),
    ("pivot_index", []),
    ("pivot_columns", []),
    ("pivot_value", []),
    ("pivot_aggs", {}),
    ("pivot_percentage", "None"),
    ("query_executed", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Step 1: Database Selection
with st.container():
    
    st.markdown('<div class="step-header">🗄️ Step 1: Select Database</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_db = st.selectbox(
            "Choose your database:",
            databases,
            index=databases.index(st.session_state.selected_db) if st.session_state.selected_db in databases else 0,
            help="Select the database you want to query"
        )
        st.session_state.selected_db = selected_db
    
    with col2:
        if selected_db:
            st.markdown('<div class="success-message">✅ Database Connected</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Step 2: Table Selection
if selected_db:
    with st.container():
        #st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown('<div class="step-header"> Step 2: Select Tables</div>', unsafe_allow_html=True)
        
        tables = get_tables(selected_db)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.session_state.selected_tables = st.multiselect(
                "Choose tables to analyze:",
                options=tables,
                default=st.session_state.selected_tables,
                help="Select one or more tables (recommended: max 5 for optimal performance)"
            )
            
            if len(st.session_state.selected_tables) > 5:
                st.warning("⚠️ Selecting more than 5 tables may impact performance")
            elif len(st.session_state.selected_tables) > 0:
                st.success(f"✅ {len(st.session_state.selected_tables)} table(s) selected")
        
        with col2:
            if st.session_state.selected_tables:
                st.markdown('<div class="info-box"><strong>Selected Tables:</strong><br>' + 
                           '<br>'.join([f"• {table}" for table in st.session_state.selected_tables]) + 
                           '</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Query Input
    if st.session_state.selected_tables:
        with st.container():
            #st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.markdown('<div class="step-header">💡 Step 3: Ask Your Question</div>', unsafe_allow_html=True)
            
            st.session_state.user_prompt = st.text_area(
                "Enter your question in plain English:",
                value=st.session_state.user_prompt,
                height=100,
                placeholder="Example: Show me total sales by month for the last year...",
                help="Describe what data you want to see from your selected tables"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 4: Execute Query
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            query_button_disabled = not (st.session_state.user_prompt.strip() and st.session_state.selected_tables)
            
            if st.button("🚀 Generate & Run Query", disabled=query_button_disabled, use_container_width=True):
                with st.spinner("🔄 Generating SQL query and fetching results..."):
                    try:
                        schema_info, schema_dict = get_schema_details(selected_db, st.session_state.selected_tables)
                        embeddings_dict = get_schema_embeddings(schema_dict)
                        matches = semantic_match(st.session_state.user_prompt, embeddings_dict, top_n=3)
                        relevant_matches = [match[0] for match in matches]
                        
                        with st.expander("🎯 Semantic Matches (Click to expand)", expanded=False):
                            st.info("Top matching columns for your query:\n" + 
                                   "\n".join([f"• {col} (score: {score:.2f})" for col, score in matches]))
                        
                        sql_query_str = prompt_to_sql_openai(st.session_state.user_prompt, st.session_state.selected_tables, schema_info, relevant_matches)
                        st.session_state.sql_query = sql_query_str
                        
                        with st.expander("📝 Generated SQL Query", expanded=False):
                            st.code(sql_query_str, language="sql")
                        
                        columns, rows = run_query(selected_db, sql_query_str)
                        if rows:
                            df = pd.DataFrame(rows, columns=columns)
                            st.session_state.query_result_df = df
                            st.session_state.query_executed = True
                            st.success("✅ Query executed successfully!")
                        else:
                            st.info("ℹ️ Query executed but returned no data.")
                            st.session_state.query_result_df = pd.DataFrame()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.session_state.query_result_df = pd.DataFrame()

# Display Results Section
if not st.session_state.query_result_df.empty:
    df = st.session_state.query_result_df
    
    st.markdown("---")
    st.markdown("## 📊 Query Results")
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Total Rows", len(df))
    with col2:
        st.metric("📊 Total Columns", len(df.columns))
    with col3:
        st.metric("💾 Data Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Display data
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="query_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="QueryResults")
        excel_data = output.getvalue()
        st.download_button(
            label="📥 Download as Excel",
            data=excel_data,
            file_name="query_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Pivot Table Section
    st.markdown('<div class="pivot-container">', unsafe_allow_html=True)
    st.markdown("## 🔄 Create Advanced Pivot Table")
    
    cols = df.columns.tolist()
    
    # Pivot configuration in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📍 Rows")
        st.session_state.pivot_index = st.multiselect(
            "Select row fields:",
            options=cols,
            default=st.session_state.pivot_index,
            help="These will become the row headers of your pivot table"
        )
    
    with col2:
        st.markdown("### 📊 Columns")
        st.session_state.pivot_columns = st.multiselect(
            "Select column fields:",
            options=cols,
            default=st.session_state.pivot_columns,
            help="These will become the column headers of your pivot table"
        )
    
    with col3:
        st.markdown("### 🔢 Values")
        st.session_state.pivot_value = st.multiselect(
            "Select value fields:",
            options=cols,
            default=st.session_state.pivot_value,
            help="These will be aggregated in the pivot table"
        )
    
    # Aggregation settings
    if st.session_state.pivot_value:
        st.markdown("### ⚙️ Aggregation Settings")
        
        agg_choices = ["sum", "mean", "count", "min", "max", "nunique"]
        if not st.session_state.pivot_aggs or set(st.session_state.pivot_aggs.keys()) != set(st.session_state.pivot_value):
            st.session_state.pivot_aggs = {val: "sum" for val in st.session_state.pivot_value}
        
        # Create columns for aggregation functions
        agg_cols = st.columns(min(len(st.session_state.pivot_value), 3))
        for i, val in enumerate(st.session_state.pivot_value):
            with agg_cols[i % 3]:
                agg = st.selectbox(
                    f"Aggregation for {val}:",
                    options=agg_choices,
                    index=agg_choices.index(st.session_state.pivot_aggs.get(val, "sum")),
                    key=f"aggfun_{val}",
                )
                st.session_state.pivot_aggs[val] = agg
    
    # Percentage options
    col1, col2 = st.columns([2, 1])
    with col1:
        st.session_state.pivot_percentage = st.selectbox(
            "Show as percentage:",
            options=["None", "Row %", "Column %", "Overall %"],
            index=["None", "Row %", "Column %", "Overall %"].index(st.session_state.pivot_percentage),
            help="Convert values to percentages based on selected calculation"
        )
    
    # Generate pivot table
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔄 Generate Pivot Table", use_container_width=True):
            if not st.session_state.pivot_index and not st.session_state.pivot_columns:
                st.warning("⚠️ Please select at least one row or column field for the pivot table.")
            elif not st.session_state.pivot_value:
                st.warning("⚠️ Please select at least one value field for aggregation.")
            else:
                with st.spinner("Creating pivot table..."):
                    pivot_df = generate_pivot_table(
                        df,
                        st.session_state.pivot_index,
                        st.session_state.pivot_columns,
                        st.session_state.pivot_aggs,
                        st.session_state.pivot_percentage,
                    )
                    
                    st.markdown("### 📋 Pivot Table Results")
                    st.dataframe(pivot_df, use_container_width=True)
                    
                    # Download pivot table
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_pivot = pivot_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Pivot as CSV",
                            data=csv_pivot,
                            file_name="pivot_table.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        excel_io = io.BytesIO()
                        with pd.ExcelWriter(excel_io, engine="openpyxl") as writer:
                            pivot_df.to_excel(writer, index=False, sheet_name="PivotTable")
                        st.download_button(
                            label="📥 Download Pivot as Excel",
                            data=excel_io.getvalue(),
                            file_name="pivot_table.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with helpful information
with st.sidebar:
    st.markdown("## 📖 Quick Guide")
    st.markdown("""
    ### How to use:
    1. **Select Database** - Choose your data source
    2. **Pick Tables** - Select relevant tables (max 5 recommended)
    3. **Ask Question** - Write what you want to know in plain English
    4. **Review Results** - Check the generated SQL and data
    5. **Create Pivot** - Build custom summaries and analyses
    
    ### Example Questions:
    - "Show total sales by month"
    - "List top 10 customers by loan amount"
    - "Compare performance across regions"
    - "Find average loan amount "
    """)
    
    if st.session_state.selected_db:
        st.markdown(f"**📍 Current Database:** `{st.session_state.selected_db}`")
    
    if st.session_state.selected_tables:
        st.markdown(f"**📋 Tables Selected:** {len(st.session_state.selected_tables)}")
