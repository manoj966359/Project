import streamlit as st

def load_styles():
    
    st.markdown("""
    <style>
    
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .main-header {
        font-size: 1.75rem;
        font-weight: 600;                             
        max-width: 980px;
        color: #1d1d1f;                             
        text-align: center;                          
        margin: 2rem auto;
        padding: 1.5rem 2rem;
        background: rgba(255, 255, 255, 0.8);        
        backdrop-filter: blur(20px);                 
        -webkit-backdrop-filter: blur(20px);
        border: none;                                 
        border-radius: 18px;                         
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); 
        letter-spacing: -0.022em;                     
        line-height: 1.2;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-sizing: border-box;
    }

    .main-header:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }

    
    .step-container {
        background: rgba(248, 248, 248, 0.8);        
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem auto;
        max-width: 980px;
        border: none;                               
        box-shadow: 0 2px 16px rgba(0, 0, 0, 0.04);
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    .step-container:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
    }

   
    .step-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1d1d1f;                              
        margin-bottom: 1.5rem;
        letter-spacing: -0.022em;
        line-height: 1.2;
    }

   
    .success-message {
    background: rgba(52, 199, 89, 0.12);    
    color: #1d1d1f;                         
    padding: 1rem 1.5rem;
    border-radius: 20px;                     
    border: 1px solid rgba(52, 199, 89, 0.2);
    margin: 1rem 0;
    font-weight: 500;                        
    font-size: 1rem;                       
    backdrop-filter: blur(24px);            
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 8px 32px rgba(52, 199, 89, 0.12);   
    transition: box-shadow 0.3s ease;
}

    .info-box {
        background: rgba(0, 122, 255, 0.08);          
        color: #1d1d1f;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 122, 255, 0.15);
        margin: 1rem 0;
        font-weight: 400;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }

    
    .pivot-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        padding: 3rem;
        border-radius: 20px;
        margin: 3rem auto;
        max-width: 1200px;
        border: none;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    .pivot-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    
    .stButton > button {
        background: #007AFF !important;             
        color: white !important;
        border: none !important;                      
        border-radius: 12px !important;               
        padding: 0.875rem 1.5rem !important;
        font-weight: 600 !important;                 
        font-size: 1rem !important;
        letter-spacing: -0.022em !important;          
        text-transform: none !important;             
        text-shadow: none !important;                 
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        cursor: pointer !important;
        min-height: 44px !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
    }

    .stButton > button:hover {
        background: #0051D5 !important;             
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(0, 81, 213, 0.3) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 1px 4px rgba(0, 122, 255, 0.3) !important;
    }

   
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        letter-spacing: -0.022em !important;
        color: #1d1d1f !important;
    }

    p, div, span {
        color: #1d1d1f !important;
        line-height: 1.47059 !important;              
    }

    
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }

    .stTextArea > div > div > textarea {
        border-radius: 8px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }

    .stMultiSelect > div > div {
        border-radius: 8px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header"> Metatalk - Text to SQL</h1>', unsafe_allow_html=True)

def render_database_selection(databases, session_state):
    """Render database selection UI"""
    with st.container():
        st.markdown('<div class="step-header"> Step 1: Select Database</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_db = st.selectbox(
                "Select your database:",
                databases,
                index=databases.index(session_state.selected_db) if session_state.selected_db in databases else 0,
                help="Select the database you want to query"
            )
            session_state.selected_db = selected_db
        
        with col2:
            if selected_db:
                st.markdown('<div class="success-message"> Database Connected</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return selected_db

def render_table_selection(tables, session_state):
    """Render table selection UI"""
    with st.container():
        st.markdown('<div class="step-header"> Step 2: Select Tables</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            session_state.selected_tables = st.multiselect(
                "Select tables to analyze:",
                options=tables,
                default=session_state.selected_tables,
                help="Select one or more tables (recommended: max 5 for optimal performance)"
            )
            
        
            if len(session_state.selected_tables) > 5:
                st.warning("⚠️ Selecting more than 5 tables may impact performance")
            elif len(session_state.selected_tables) > 0:
                st.success(f"✅ {len(session_state.selected_tables)} table(s) selected")
        
        with col2:
            if session_state.selected_tables:
                st.markdown('<div class="info-box"><strong>Selected Tables:</strong><br>' + 
                           '<br>'.join([f"• {table}" for table in session_state.selected_tables]) + 
                           '</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_query_input(session_state):
    """Render query input UI"""
    with st.container():
        st.markdown('<div class="step-header">💡 Step 3: Ask Your Question</div>', unsafe_allow_html=True)
        
        session_state.user_prompt = st.text_area(
            "Enter your question in plain English:",
            value=session_state.user_prompt,
            height=100,
            placeholder="Example: Show me total sales by month for the last year...",
            help="Describe what data you want to see from your selected tables"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_results_section(df):
    """Render results display section"""
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

def render_download_buttons(df):
    """Render CSV and Excel download buttons"""
    import io
    import pandas as pd
    
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

def render_pivot_section(df, session_state):
    """Render pivot table configuration section"""
    st.markdown('<div class="pivot-container">', unsafe_allow_html=True)
    st.markdown("## 🔄 Create Advanced Pivot Table")
    
    cols = df.columns.tolist()
    
    # Pivot configuration in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📍 Rows")
        session_state.pivot_index = st.multiselect(
            "Select row fields:",
            options=cols,
            default=session_state.pivot_index,
            help="These will become the row headers of your pivot table"
        )
    
    with col2:
        st.markdown("### 📊 Columns")
        session_state.pivot_columns = st.multiselect(
            "Select column fields:",
            options=cols,
            default=session_state.pivot_columns,
            help="These will become the column headers of your pivot table"
        )
    
    with col3:
        st.markdown("### 🔢 Values")
        session_state.pivot_value = st.multiselect(
            "Select value fields:",
            options=cols,
            default=session_state.pivot_value,
            help="These will be aggregated in the pivot table"
        )
    
    return cols

def render_aggregation_settings(session_state):
    """Render aggregation settings for pivot table"""
    if session_state.pivot_value:
        st.markdown("### ⚙️ Aggregation Settings")
        
        agg_choices = ["sum", "mean", "count", "min", "max", "nunique"]
        if not session_state.pivot_aggs or set(session_state.pivot_aggs.keys()) != set(session_state.pivot_value):
            session_state.pivot_aggs = {val: "sum" for val in session_state.pivot_value}
        
        # Create columns for aggregation functions
        agg_cols = st.columns(min(len(session_state.pivot_value), 3))
        for i, val in enumerate(session_state.pivot_value):
            with agg_cols[i % 3]:
                agg = st.selectbox(
                    f"Aggregation for {val}:",
                    options=agg_choices,
                    index=agg_choices.index(session_state.pivot_aggs.get(val, "sum")),
                    key=f"aggfun_{val}",
                )
                session_state.pivot_aggs[val] = agg

def render_percentage_options(session_state):
    """Render percentage options for pivot table"""
    col1, col2 = st.columns([2, 1])
    with col1:
        session_state.pivot_percentage = st.selectbox(
            "Show as percentage:",
            options=["None", "Row %", "Column %", "Overall %"],
            index=["None", "Row %", "Column %", "Overall %"].index(session_state.pivot_percentage),
            help="Convert values to percentages based on selected calculation"
        )

def render_pivot_download_buttons(pivot_df):
    """Render download buttons for pivot table"""
    import io
    import pandas as pd
    
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

def render_sidebar(session_state):
    """Render sidebar with helpful information"""
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
        
        if session_state.selected_db:
            st.markdown(f"**📍 Current Database:** `{session_state.selected_db}`")
        
        if session_state.selected_tables:
            st.markdown(f"**📋 Tables Selected:** {len(session_state.selected_tables)}")
