import streamlit as st
import pandas as pd
import plotly.express as px

def render_dashboard():
    """Renders the right panel dashboard and SQL viewer."""
    
    # We use tabs to keep the right panel clean
    tab_data, tab_analytics, tab_kpi = st.tabs(["Live Data View", "Analytics & Insights", "Executive Summary and insight"])
    
    with tab_data:
        st.markdown("### Tabular Data")
        
        # Check if there are live query results from the agent
        if "current_sql_results" in st.session_state and st.session_state.current_sql_results:
            st.success("Showing live filtered results from your last chat query:")
            
            try:
                # Convert list of dicts to DataFrame for beautiful display
                df_results = pd.DataFrame(st.session_state.current_sql_results)
                st.dataframe(df_results, use_container_width=True, height=300)
            except Exception as e:
                st.error("Could not display results as a table.")
                st.write(st.session_state.current_sql_results)
                
        # If no query results yet, show the raw uploaded tables
        elif "db_manager" in st.session_state and st.session_state.db_manager.tables:
            st.info("Showing raw uploaded data. Ask the chat to filter or analyze it!")
            for table_name in st.session_state.db_manager.tables.keys():
                with st.expander(f"Table: {table_name}", expanded=True):
                    try:
                        # Fetch top 100 rows for preview to keep it fast
                        preview_query = f"SELECT * FROM {table_name} LIMIT 100"
                        df_preview = st.session_state.db_manager.execute_query(preview_query)
                        st.dataframe(df_preview, use_container_width=True, height=300)
                    except Exception as e:
                        st.error(f"Could not load preview for {table_name}: {e}")
        else:
            st.warning("No data uploaded yet. Please upload an Excel or CSV file in the sidebar.")
            
            
    with tab_analytics:
        # Business Insights
        st.markdown("#### Insights of Question")
        if "insights" in st.session_state and st.session_state.insights:
            st.markdown(st.session_state.insights)
        else:
            st.write("Detailed insights of the question will be automatically generated and displayed here.")

    with tab_kpi:
        st.markdown("### Executive Summary and insight")
        if "db_manager" in st.session_state and st.session_state.db_manager.tables:
            table_names = list(st.session_state.db_manager.tables.keys())
            selected_table = st.selectbox("Select Table for Analysis", table_names)
            
            if selected_table:
                try:
                    df_full = st.session_state.db_manager.execute_query(f"SELECT * FROM {selected_table}")
                    
                    # KPIs
                    st.markdown("#### Table Overview")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Rows", len(df_full))
                    col2.metric("Total Columns", len(df_full.columns))
                    numeric_cols = df_full.select_dtypes(include='number').columns.tolist()
                    col3.metric("Numeric Columns", len(numeric_cols))
                    
                    st.divider()
                    st.markdown("#### Auto-Generated Data Summary & Insights")
                    
                    if "table_summaries" not in st.session_state:
                        st.session_state.table_summaries = {}
                        
                    if selected_table not in st.session_state.table_summaries:
                        gemini_key = st.session_state.get("gemini_key")
                        if gemini_key:
                            with st.spinner(f"Generating summary for {selected_table}..."):
                                from langchain_google_genai import ChatGoogleGenerativeAI
                                from langchain_core.messages import SystemMessage, HumanMessage
                                
                                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=gemini_key)
                                preview = df_full.head(10).to_string()
                                prompt = f"Analyze the following data sample from the uploaded table '{selected_table}'. Provide an executive summary and key insights that are important. Keep it concise but comprehensive.\n\nData Sample:\n{preview}"
                                
                                try:
                                    summary = llm.invoke([
                                        SystemMessage(content="You are an expert data analyst."),
                                        HumanMessage(content=prompt)
                                    ]).content
                                    st.session_state.table_summaries[selected_table] = summary
                                except Exception as e:
                                    st.error(f"Error generating summary: {e}")
                        else:
                            st.info("Please configure your Gemini API Key in the sidebar to automatically generate the executive summary.")
                            
                    if selected_table in st.session_state.table_summaries:
                        st.markdown(st.session_state.table_summaries[selected_table])
                        
                except Exception as e:
                    st.error(f"Error loading table data: {e}")
        else:
            st.warning("No data uploaded yet. Please upload an Excel or CSV file in the sidebar to view KPIs and Insights.")
