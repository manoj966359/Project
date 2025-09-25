import streamlit as st
import pandas as pd
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from logic import (
    get_databases, get_tables, get_schema_details, get_schema_embeddings,
    semantic_match, prompt_to_sql_openai, run_query
)
from pivot import generate_pivot_table
from ui_components import (
    load_styles, render_header, render_database_selection,
    render_table_selection, render_query_input, render_sidebar
)
 
st.set_page_config(
    page_title="Metatalk - Text to SQL",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# Load styles
load_styles()
 
# Render header
render_header()
 
# Get databases
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
selected_db = render_database_selection(databases, st.session_state)
 
# Step 2: Table Selection
if selected_db:
    tables = get_tables(selected_db)
    render_table_selection(tables, st.session_state)
 
    # Step 3: Query Input
    if st.session_state.selected_tables:
        render_query_input(st.session_state)
       
        # Step 4: Execute Query
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            query_button_disabled = not (st.session_state.user_prompt.strip() and st.session_state.selected_tables)
           
            if st.button("Generate & Run Query", disabled=query_button_disabled, use_container_width=True):
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
 
# Render sidebar
render_sidebar(st.session_state)
 
# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)
