import os
from dotenv import load_dotenv
from langsmith import evaluate, Client
from langsmith.schemas import Run, Example
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if not os.environ.get("LANGCHAIN_API_KEY") or not os.environ.get("GOOGLE_API_KEY"):
    print("❌ Error: Please ensure LANGCHAIN_API_KEY and GOOGLE_API_KEY are in your .env file.")
    exit(1)

try:
    from agents.workflow import workflow as app
    if app is None:
        from agents.workflow import build_graph
        app = build_graph()
except ImportError:
    from agents.workflow import build_graph
    app = build_graph()

client = Client()
dataset_name = "BusinessIQ-Golden-Dataset"

# ==========================================
# 1. APP INVOCATION (Now grabs Context too)
# ==========================================
def predict_business_iq(inputs: dict):
    question = inputs["question"]
    state = app.invoke({"messages": [question]})
    
    actual_output = state["messages"][-1].content
    routed_agent = state.get("current_agent", "unknown") 
    
    # Try to grab the FAISS chunks from your state. 
    # If your state doesn't save them yet, we pass a fallback message.
    context = state.get("context", "Context not found in state.")
    
    return {
        "actual_output": actual_output,
        "routed_agent": routed_agent,
        "context": context
    }

# ==========================================
# 2. THE 4 METRICS (EVALUATORS)
# ==========================================
eval_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Metric 1: Routing Accuracy
def evaluate_routing(run: Run, example: Example) -> dict:
    expected = example.outputs.get("expected_agent")
    actual = run.outputs.get("routed_agent")
    return {"key": "routing_accuracy", "score": 1 if expected == actual else 0}

# Metric 2: Correctness (QA)
def evaluate_correctness(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("actual_output", "")
    reference = example.outputs.get("expected_output", "")
    prompt = f"Question: {example.inputs.get('question')}\nExpected: {reference}\nActual: {prediction}\nDoes the Actual answer capture the meaning of the Expected answer? Reply ONLY '1' for yes, '0' for no."
    try:
        score = int(eval_llm.invoke(prompt).content.strip()) if eval_llm.invoke(prompt).content.strip() in ['0', '1'] else 0
    except: score = 0
    return {"key": "correctness", "score": score}

# Metric 3: Faithfulness (Hallucination Check)
def evaluate_faithfulness(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("actual_output", "")
    context = run.outputs.get("context", "")
    prompt = f"Context: {context}\nPrediction: {prediction}\nIs all the information in the Prediction strictly derived from the Context without making up new facts? Reply ONLY '1' for yes (faithful), '0' for no (hallucination)."
    try:
        score = int(eval_llm.invoke(prompt).content.strip()) if eval_llm.invoke(prompt).content.strip() in ['0', '1'] else 0
    except: score = 0
    return {"key": "faithfulness", "score": score}

# Metric 4: Answer Relevance
def evaluate_relevance(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("actual_output", "")
    question = example.inputs.get("question", "")
    prompt = f"Question: {question}\nPrediction: {prediction}\nDoes the Prediction directly and usefully answer the Question without going off-topic? Reply ONLY '1' for yes, '0' for no."
    try:
        score = int(eval_llm.invoke(prompt).content.strip()) if eval_llm.invoke(prompt).content.strip() in ['0', '1'] else 0
    except: score = 0
    return {"key": "answer_relevance", "score": score}

# ==========================================
# 3. RUN EVALUATION
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Advanced LangSmith Evaluation...")
    experiment_results = evaluate(
        predict_business_iq,
        data=dataset_name, 
        evaluators=[evaluate_routing, evaluate_correctness, evaluate_faithfulness, evaluate_relevance],
        experiment_prefix="business-iq-v2-advanced",
    )
    import time
    time.sleep(2) # Prevent thread shutdown error
    print("\n✅ Evaluation complete! Check LangSmith dashboard.")