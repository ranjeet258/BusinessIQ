from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage
from data.db_manager import DuckDBManager
from core.state import AgentState

def create_sql_node(db_manager: DuckDBManager, google_api_key: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key)
    
    def sql_node(state: AgentState):
        query = state["messages"][-1].content
        schema = db_manager.get_schema()
        
        system_msg = SystemMessage(
            content=f"You are a SQL expert. Schema:\n{schema}\n"
                    "IMPORTANT: You have access to multiple tables. Ensure you query the correct table(s) for the data requested, and use JOINs if the question requires data from multiple tables.\n"
                    "Generate ONLY valid DuckDB SQL to answer the user query. Do not include markdown formatting or explanations."
        )
        
        sql_query = llm.invoke([system_msg] + state["messages"]).content.strip().replace("```sql", "").replace("```", "")
        
        try:
            results_df = db_manager.execute_query(sql_query)
            results_dict = results_df.to_dict(orient="records")
            response = f"SQL Executed successfully. Found {len(results_dict)} rows."
            return {
                "current_sql_query": sql_query,
                "current_sql_results": results_dict,
                "messages": [SystemMessage(content=response)],
                "current_agent": "sql"
            }
        except Exception as e:
            return {
                "current_sql_query": sql_query,
                "messages": [AIMessage(content=f"Generated SQL: {sql_query}")],
                "current_agent": "sql"
            }
            
    return sql_node
