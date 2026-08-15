import os
import math
import time
import random
import urllib.parse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from database import save_memory, search_memory
from rag import retrieve_from_rag

Path("uploads").mkdir(exist_ok=True)

load_dotenv(override=True)


CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


if os.getenv("TAVILY_API_KEY"):
    from langchain_tavily import TavilySearch
    web_search = TavilySearch(
        max_results=5,
        topic="general",
        search_depth="advanced"
    )
else:
    @tool
    def web_search(query: str) -> str:
        """
        Search the web for latest/current information.
        """
        return "Tavily API key is not configured in .env file. Please set TAVILY_API_KEY to enable live web search."


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"
    


@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents.
    """

    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )




@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )



@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )


def enhance_image_prompt(user_prompt: str) -> str:
    """
    Uses Gemini LLM to convert the user's request into a master-level image generation prompt.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return user_prompt

    try:
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL") or "gemini-3.5-flash",
            google_api_key=api_key,
            temperature=0.75
        )

        prompt_system = f"""You are an elite AI image prompt engineer for state-of-the-art image models (like DALL-E 3, Midjourney v6, and Flux).

Convert the user's request into a single master-level, breathtaking image generation prompt.

Guidelines for elite prompt expansion:
- Add a compelling focal subject / perspective (e.g. a character or vantage point on a high balcony/rooftop overlooking the scene).
- Add rich environmental and atmospheric details (e.g. towering skyscrapers, glowing neon holographic billboards, Japanese kanji signs, flying vehicles with light streaks, dramatic fiery sunset clouds, reflections on wet surfaces).
- Add vibrant color harmony (e.g. electric cyan, neon magenta, deep violet, golden amber sunset glow).
- Add photorealistic rendering tags: (e.g. cinematic photography, 8k resolution, masterpiece, ray tracing, Unreal Engine 5 render, sharp focus, volumetric lighting, ultra-detailed).
- Return ONLY the prompt text. Do not add quotes, introductory phrases, headings, or markdown.

User Request: {user_prompt}"""

        res = llm.invoke(prompt_system)
        enhanced = res.content.strip() if hasattr(res, "content") else str(res).strip()
        if enhanced.startswith('"') and enhanced.endswith('"'):
            enhanced = enhanced[1:-1].strip()
        return enhanced if enhanced else user_prompt
    except Exception as e:
        print(f"Prompt enhancement warning: {e}")
        return user_prompt


@tool
def generate_image(prompt: str) -> str:
    """
    Generate high-quality AI images, photos, illustrations, and artwork from a text prompt.
    Use this tool whenever the user asks to generate, create, draw, visualize, make, or produce an image or picture.
    """
    try:
        # 1. Enhance the prompt using LLM for DALL-E 3 / Flux quality
        enhanced_prompt = enhance_image_prompt(prompt)
        
        # 2. Build URL for Pollinations AI
        seed = random.randint(100000, 999999)
        encoded_prompt = urllib.parse.quote(enhanced_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&seed={seed}&nologo=true"

        # 3. Save locally to uploads/ for permanent caching and instant loading
        filename = f"image-{int(time.time())}-{seed}.png"
        file_path = Path("uploads") / filename
        
        display_url = image_url
        try:
            req = urllib.request.Request(
                image_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.read())
                    display_url = f"/uploads/{filename}"
        except Exception as dl_err:
            print(f"Local save notice: {dl_err}, using direct URL.")
            display_url = image_url

        # Clean ChatGPT-style markdown image output
        return f"![{prompt}]({display_url})"
    except Exception as e:
        return f"❌ Failed to generate image: {str(e)}"


tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search,
    generate_image
]