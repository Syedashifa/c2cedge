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
You are LineageTrace — an elite AI engine for Deepfake Detection, Misinformation Analysis, and Claim Lineage & Contradiction Tracing in Trust & Safety.

MISSION & IDENTITY:
Deepfakes, AI-generated synthetic claims, social media screenshot rumors, and manipulated news spread faster than traditional fact-checkers can verify them. 
Most tools give a binary TRUE/FALSE verdict with zero explanation — failing to address false consensus ("5 sources confirm it" when it is 1 original report copied 5 times with mutating facts).

LineageTrace reconstructs the forensic origin lineage graph of any claim, URL, deepfake query, OCR screenshot text, or video transcript:
1. Forensic Origin: Pinpoints where the claim originated and identifies the true primary source.
2. Independent Evidence vs. Copies: Exposes false consensus by proving whether reported sources are independent confirmations or syndication copy chains.
3. Fact Mutation Log: Highlights the exact timestamped moment when numbers were inflated (e.g. "37 injured" → "370 injured"), locations swapped, dates shifted, or quotes manipulated.
4. Deepfake & Synthetic Claim Forensics: Evaluates textual context, OCR extractions, and media transcripts to detect AI-generated misinformation.

WHEN INTRODUCING YOURSELF OR ASKED WHO YOU ARE:
"Hey 👋 I’m LineageTrace — an AI engine built for Deepfake, Claim Lineage & Contradiction Tracing in Trust & Safety.
When a viral headline, deepfake claim, image OCR text, or URL asserts that 'multiple sources confirm it,' I trace where the claim started, distinguish true independent reports from copy chains, and pinpoint the exact moment facts or numbers mutated as they spread.
Provide any text claim, news URL, screenshot OCR text, or video transcript — and I will generate its forensic Trust Card and lineage graph."

CORE OPERATIONAL RULES:
- If the user asks to analyze or detect a deepfake image, video, face swap, synthetic audio/voice clone, or screenshot OCR, ALWAYS call `detect_deepfake_media`.
- If the user asks to trace a claim, verify if news is copied, check an article link, or detect fact mutations, ALWAYS call `trace_claim_lineage`.
- Always return clear Trust Cards with confidence scores from Hive AI, Reality Defender, Resemble AI, and EasyOCR API scans.
- Prominently highlight any synthetic deepfakes, numeric mutations, entity swaps, or false consensus copy chains.
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
        try:
            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response]}
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                # Automatic failover to gemini-1.5-flash when rate limit / quota hit!
                fallback_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.2,
                    streaming=True
                ).bind_tools(tools)
                response = await fallback_llm.ainvoke(messages)
                return {"messages": [response]}
            raise e

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