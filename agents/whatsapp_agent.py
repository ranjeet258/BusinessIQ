import os
import requests
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.state import AgentState
from dotenv import load_dotenv

load_dotenv()

def create_whatsapp_node(google_api_key: str = "", grok_api_key: str = "", hf_key: str = ""):
    if grok_api_key:
        llm = ChatOpenAI(model="grok-beta", api_key=grok_api_key, base_url="https://api.x.ai/v1")
    elif google_api_key:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=google_api_key)
    elif hf_key:
        hf_endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-72B-Instruct",
            huggingfacehub_api_token=hf_key,
            task="text-generation",
            max_new_tokens=1024
        )
        llm = ChatHuggingFace(llm=hf_endpoint)
    else:
        raise ValueError("No LLM API key provided")
    
    def whatsapp_node(state: AgentState):
        query = state["messages"][-1].content
        insights = state.get("insights") or "No recent insights available to send."
        
        # 1. Extract phone number and reframe message
        system_prompt = f"""You are a world-class marketing copywriter integrated with WhatsApp.
The user wants to send a promotional advertisement to a specific WhatsApp number based on recent data insights.
Extract the target phone number from the user's request (ensure it has a country code, e.g., +1234567890). If no country code is provided, assume +91 (India) as default or infer from context.

Recent Insights/Data to reframe:
{insights}

CRITICAL INSTRUCTION FOR REFRAMED MESSAGE:
Your job is to take the provided insights and turn them into a REAL, HIGH-CONVERTING WhatsApp advertisement. 
It MUST NOT look like a boring small paragraph. It MUST look like a professional, exciting promotional message.
Ensure the advertisement includes:
- An attention-grabbing headline with emojis (e.g. 🚨 🌟 🔥)
- Bullet points to highlight key benefits, offers, or data points
- Strong persuasive and engaging language
- A clear Call to Action (CTA) at the end (e.g., "Reply YES to claim!", "Contact us today!")
- Proper spacing and line breaks for easy readability on a mobile phone

Return strictly a JSON object with two keys:
1. "phone_number": The extracted phone number as a string (with + and country code).
2. "reframed_message": The highly formatted, exciting advertisement message to send.
Return ONLY valid JSON."""

        messages_for_llm = [SystemMessage(content=system_prompt)] + state["messages"]
        
        try:
            resp = llm.invoke(messages_for_llm).content.strip()
            if resp.startswith("```json"):
                resp = resp[7:]
            if resp.endswith("```"):
                resp = resp[:-3]
                
            parsed_resp = json.loads(resp.strip())
            phone_number = parsed_resp.get("phone_number", "")
            reframed_message = parsed_resp.get("reframed_message", "")
            
            # Clean phone number (remove spaces, dashes, and leading +)
            phone_number = re.sub(r'[^\d]', '', phone_number)
            
            if not phone_number:
                return {
                    "messages": [AIMessage(content=reframed_message or "I couldn't detect a valid phone number in your request. Please specify the target number.")],
                    "current_agent": "whatsapp"
                }
                
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error understanding the WhatsApp request: {e}")],
                "current_agent": "whatsapp"
            }
            
        # 2. Send via Meta WhatsApp Cloud API
        try:
            import streamlit as st
            wa_token = st.session_state.get("wa_token") if hasattr(st, "session_state") else None
            wa_phone_id = st.session_state.get("wa_phone_id") if hasattr(st, "session_state") else None
        except Exception:
            wa_token = None
            wa_phone_id = None

        access_token = wa_token or os.environ.get("WHATSAPP_ACCESS_TOKEN")
        phone_number_id = wa_phone_id or os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
        
        if not access_token or not phone_number_id:
            return {
                "messages": [AIMessage(content=f"WhatsApp draft prepared for +{phone_number}:\n\n{reframed_message}\n\n(Note: WhatsApp API credentials not configured in .env)")],
                "current_agent": "whatsapp"
            }
            
        url = f"https://graph.facebook.com/v25.0/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "text",
            "text": {
                "body": reframed_message
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return {
                "messages": [AIMessage(content=f"Successfully sent the advertisement to {phone_number}!\n\n**Message sent:**\n{reframed_message}")],
                "current_agent": "whatsapp"
            }
        except requests.exceptions.RequestException as e:
            err_msg = str(e)
            if hasattr(e, "response") and e.response is not None and e.response.text:
                err_msg += f" | Details: {e.response.text}"
            return {
                "messages": [AIMessage(content=f"Failed to send WhatsApp message to {phone_number}. Error: {err_msg}")],
                "current_agent": "whatsapp"
            }

    return whatsapp_node
