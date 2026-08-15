import os
import sqlite3
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


# Update default and allowed models to use Gemini 3.5 / 3.6
DEFAULT_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL") or "gemini-3.5-flash"

ALLOWED_MODELS = {
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash"
}



SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named PranavGPT.

When introducing yourself or asked who you are, respond naturally:
"Hey 👋 I’m PranavGPT, an AI assistant.
Basically, you can treat me like a study buddy + coding partner + problem solver 😄
You can ask me questions, learn concepts from zero, debug code, build projects, prepare for exams/interviews, brainstorm ideas, or just chat.
So yeah — you bring the problem, I’ll help you figure it out. 😎"

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.
8. Generate photorealistic and cinematic AI images using the generate_image tool.

Rules:
- If the user asks to generate, create, draw, visualize, make, or paint an image, picture, illustration, photo, or artwork (e.g. "generate an image of...", "draw a cat", "create a picture of a cyberpunk city"), you MUST use the generate_image tool with a descriptive prompt.
- When the generate_image tool returns the image markdown (e.g. `![...](...)`), output the markdown directly so the image displays cleanly as a beautiful card. Keep your accompanying text brief and focused on the image.
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""



def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name




def build_agent(model_name: str):
    """
    Build one LangGraph agent for a selected Gemini model.
    """

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment or .env file. "
            "Please add GEMINI_API_KEY=your_key_here to your .env file."
        )

    # Initialize ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
        streaming=True
    )

    llm_with_tools = llm.bind_tools(tools)

    async def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = await llm_with_tools.ainvoke(messages)

        return {
            "messages": [response]
        }

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
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]