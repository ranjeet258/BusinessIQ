from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.state import AgentState
import pandas as pd
import json

def create_analysis_node(gemini_key: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=gemini_key)
    
    def analysis_node(state: AgentState):
        results = state.get("current_sql_results", [])
        if not results:
            return state
            
        system_prompt = f"""You are an expert business analyst. You will be provided with the conversation history and the resulting data from the latest query.
Data Results:
{str(results[:10])}

You must return your analysis strictly as a JSON object with exactly two keys:
1. "chat_answer": A concise 2-4 lines sentence summarizing the answer to the query based on the results.
2. "detailed_analysis": A deep, detailed analysis of the data, and include at the end 2-3 similar or related things they can ask about this data.
Return ONLY valid JSON, no markdown formatting like ```json."""

        messages_for_llm = [SystemMessage(content=system_prompt)] + state["messages"]
        
        try:
            insights_resp = llm.invoke(messages_for_llm).content.strip()
            if insights_resp.startswith("```json"):
                insights_resp = insights_resp[7:]
            if insights_resp.endswith("```"):
                insights_resp = insights_resp[:-3]
            
            parsed_resp = json.loads(insights_resp.strip())
            chat_answer = parsed_resp.get("chat_answer", "Analysis complete.")
            detailed_analysis = parsed_resp.get("detailed_analysis", str(insights_resp))
        except Exception as e:
            chat_answer = "I have analyzed the data. See the detailed insights in the dashboard."
            detailed_analysis = f"Error parsing detailed analysis: {e}\nRaw output: {insights_resp}"
        
        return {
            "insights": detailed_analysis,
            "messages": [AIMessage(content=chat_answer)]
        }
        
    return analysis_node
