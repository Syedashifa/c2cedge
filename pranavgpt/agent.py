import os
from pathlib import Path
from dotenv import load_dotenv
import certifi

# Clean environment to prevent stale cached keys
for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"]:
    if key in os.environ:
        del os.environ[key]

load_dotenv(override=True)

# Synchronize API keys so langchain uses the active one
active_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if active_key:
    os.environ["GOOGLE_API_KEY"] = active_key
    os.environ["GEMINI_API_KEY"] = active_key

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from tools import tools

Path("data").mkdir(exist_ok=True)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL") or "gemini-3.5-flash"

ALLOWED_MODELS = {
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash"
}


SYSTEM_PROMPT = """
You are LineageTrace — an AI-powered Claim Lineage & Contradiction Tracer for Misinformation and Trust & Safety tracking.

When introducing yourself or asked who you are, respond naturally:
"Hey 👋 I’m LineageTrace, a Claim Lineage & Contradiction Tracer.
Instead of giving a simplistic TRUE/FALSE verdict, I trace where claims actually started, show which sources are truly independent versus copies of an original, and highlight the exact moment facts or numbers mutated as they spread (e.g. '37 injured' becoming '370').
You can provide a text claim, news article URL, screenshot text, or video transcript — and I'll trace its lineage graph."

You can:
1. Trace Claim Lineage & Mutations: Use the `trace_claim_lineage` tool whenever the user provides a claim, rumor, link, or screenshot to trace.
2. Web Search: Search the web using Tavily Search (`web_search`).
3. Document RAG: Search uploaded files using `search_uploaded_documents`.
4. Memory: Save or recall facts using `remember_this` and `recall_memory`.
5. Image Generation: Generate illustrations using `generate_image`.

Rules:
- If the user asks to trace a claim, verify if news is copied, analyze a rumor, or inspect a URL/screenshot, ALWAYS call `trace_claim_lineage`.
- Always return the Trust Card with evidence + confidence + uncertainty.
- Highlight any numeric mutations, entity swaps, or false consensus copies clearly.
"""


def normalize_model_name(model_name: str | None) -> str:
    if not model_name:
        return DEFAULT_MODEL
    model_name = model_name.strip()
    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL
    return model_name


def build_agent(model_name: str):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment or .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        streaming=True
    )

    llm_with_tools = llm.bind_tools(tools)

    async def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    selected_model = normalize_model_name(model_name)
    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)
    return _AGENT_CACHE[selected_model]