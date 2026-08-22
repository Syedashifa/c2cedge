import os
import math
import time
import random
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from database import save_memory, search_memory
from rag import retrieve_from_rag
from lineage_engine import build_lineage_graph

Path("uploads").mkdir(exist_ok=True)

load_dotenv(override=True)

CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


if os.getenv("TAVILY_API_KEY"):
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
else:
    tavily_client = None


@tool
def web_search(query: str) -> str:
    """
    Search the web for latest/current information using Tavily Search.
    """
    if not tavily_client:
        return "Tavily API key is not configured in .env file. Please set TAVILY_API_KEY to enable live web search."

    try:
        res = tavily_client.search(query=query, search_depth="advanced", max_results=5)
        results = res.get("results", [])
        if not results:
            return "No web results found."
        
        output = []
        for r in results:
            output.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nSnippet: {r.get('content')}\n")
        return "\n---\n".join(output)
    except Exception as e:
        return f"Web search error: {str(e)}"


@tool
def trace_claim_lineage(claim_or_url: str, source_type: str = "text") -> str:
    """
    Core LineageTrace Tool:
    Traces claim origins, distinguishes independent sources vs. copy chains,
    and flags chronological fact mutations (e.g. numeric inflation, entity swaps).
    
    Inputs:
    - claim_or_url: The text claim, URL, or screenshot OCR text to trace.
    - source_type: 'text', 'url', 'image', or 'video'.
    """
    if not tavily_client:
        return "⚠️ LineageTrace search engine requires TAVILY_API_KEY set in .env."

    clean_claim = claim_or_url.strip()

    # Step 1: Query Agent - Extract text if URL or screenshot text
    if clean_claim.startswith("http://") or clean_claim.startswith("https://"):
        try:
            req = urllib.request.Request(
                clean_claim,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
                # Basic title and text extraction
                title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
                extracted_title = title_match.group(1) if title_match else ""
                clean_claim = f"{extracted_title} {clean_claim}"
        except Exception:
            pass

    # Step 2: Retrieval Agent - Search web sources
    try:
        tavily_res = tavily_client.search(
            query=clean_claim,
            search_depth="advanced",
            max_results=7
        )
        sources_raw = tavily_res.get("results", [])
    except Exception as e:
        return f"❌ LineageTrace retrieval error: {str(e)}"

    if not sources_raw:
        return (
            "🔴 **TRUST CARD: ORIGIN UNCLEAR**\n\n"
            f"> No online sources found for: *\"{clean_claim}\"*\n\n"
            "Organically spreading claims with no traceable online origin resolve to **Origin Unclear**."
        )

    # Format retrieved sources
    retrieved_sources = []
    for idx, s in enumerate(sources_raw):
        retrieved_sources.append({
            "title": s.get("title", f"Source {idx+1}"),
            "url": s.get("url", ""),
            "domain": s.get("url", "").split("/")[2] if "//" in s.get("url", "") else "web",
            "snippet": s.get("content", ""),
            "published_date": s.get("published_date", f"2026-08-22 T+{idx*2}h")
        })

    # Step 3 & 4: Lineage Agent & Mutation Agent (ML Pairwise Classifier + Mutation Detection)
    graph_res = build_lineage_graph(clean_claim, retrieved_sources)

    # Step 3b: Conditional Agentic Branching (Gray zone ambiguity triggering re-search)
    if graph_res.get("need_re_search") and graph_res.get("re_search_query"):
        try:
            re_search_res = tavily_client.search(
                query=graph_res["re_search_query"],
                search_depth="advanced",
                max_results=3
            )
            extra_sources = re_search_res.get("results", [])
            for idx, s in enumerate(extra_sources):
                retrieved_sources.append({
                    "title": s.get("title", f"Narrowed Source {idx+1}"),
                    "url": s.get("url", ""),
                    "domain": s.get("url", "").split("/")[2] if "//" in s.get("url", "") else "web",
                    "snippet": s.get("content", ""),
                    "published_date": s.get("published_date", "2026-08-22")
                })
            # Re-evaluate graph with narrowed evidence
            graph_res = build_lineage_graph(clean_claim, retrieved_sources)
        except Exception:
            pass

    # Step 5: Explainer Agent - Format output into Trust Card + Forensic Lineage Details
    trust_badge = graph_res["trust_badge"]
    summary_headline = graph_res["summary_headline"]
    ind_count = graph_res["independent_origins_count"]
    tot_count = graph_res["total_sources_count"]

    # Build Markdown Trust Card
    trust_card_md = []
    trust_card_md.append(f"### {trust_badge}")
    trust_card_md.append(f"**Claim Analysis:** *\"{clean_claim[:120]}\"*")
    trust_card_md.append(f"> **Summary:** {summary_headline}\n")
    trust_card_md.append(f"- 📊 **Independent Confirmations:** `{ind_count}` of `{tot_count}` total sources")

    if graph_res["mutations"]:
        trust_card_md.append("\n#### ⚠️ Fact Mutation Log:")
        for mut in graph_res["mutations"]:
            trust_card_md.append(f"- 🚩 **[{mut['type']}]**: {mut['description']}")

    trust_card_md.append("\n#### 🔬 Forensic Lineage Graph:")
    for edge in graph_res["edges"]:
        trust_card_md.append(
            f"- `[{edge['relationship'].upper()}]` Edge confidence: `{edge['confidence']*100:.1f}%` "
            f"(Embedding sim: `{edge['embedding_similarity']}`, 3-gram overlap: `{edge['ngram_overlap']}`)"
        )

    trust_card_md.append("\n#### 📚 Discovered Sources & Chain:")
    for node in graph_res["nodes"]:
        origin_tag = " [PRIMARY ORIGIN]" if node["is_independent_origin"] else " [COPY / AGGREGATOR]"
        trust_card_md.append(f"- **{node['title']}** ({node['domain']}){origin_tag}\n  🔗 [View Source]({node['url']})")

    # Embed JSON block for UI frontend graph rendering if needed
    json_block = json.dumps({
        "trust_status": graph_res["trust_status"],
        "independent_origins": ind_count,
        "total_sources": tot_count,
        "nodes": graph_res["nodes"],
        "edges": graph_res["edges"],
        "mutations": graph_res["mutations"]
    })

    trust_card_md.append(f"\n<!-- LINEAGE_JSON: {json_block} -->")

    return "\n".join(trust_card_md)


@tool
def calculator(expression: str) -> str:
    """Useful for simple math calculations."""
    try:
        allowed = {"math": math, "abs": abs, "round": round, "min": min, "max": max, "sum": sum}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def search_uploaded_documents(query: str) -> str:
    """Search uploaded documents for relevant information."""
    return retrieve_from_rag(query=query, thread_id=CURRENT_THREAD_ID)


@tool
def remember_this(memory: str) -> str:
    """Save an important user preference or fact into long-term memory."""
    return save_memory(thread_id=CURRENT_THREAD_ID, memory=memory)


@tool
def recall_memory(query: str) -> str:
    """Recall saved long-term memories about the user or this conversation."""
    return search_memory(thread_id=CURRENT_THREAD_ID, query=query)


def enhance_image_prompt(user_prompt: str) -> str:
    """Uses Gemini LLM to convert the user's request into a master-level image generation prompt."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return user_prompt

    try:
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL") or "gemini-3.5-flash",
            google_api_key=api_key,
            temperature=0.75
        )

        prompt_system = f"Convert into single master-level image generation prompt for photorealistic output: {user_prompt}"
        res = llm.invoke(prompt_system)
        enhanced = res.content.strip() if hasattr(res, "content") else str(res).strip()
        return enhanced if enhanced else user_prompt
    except Exception:
        return user_prompt


@tool
def generate_image(prompt: str) -> str:
    """Generate high-quality AI images, photos, illustrations, and artwork from a text prompt."""
    try:
        enhanced_prompt = enhance_image_prompt(prompt)
        seed = random.randint(100000, 999999)
        encoded_prompt = urllib.parse.quote(enhanced_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&seed={seed}&nologo=true"
        filename = f"image-{int(time.time())}-{seed}.png"
        file_path = Path("uploads") / filename

        display_url = image_url
        try:
            req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.read())
                    display_url = f"/uploads/{filename}"
        except Exception:
            display_url = image_url

        return f"![{prompt}]({display_url})"
    except Exception as e:
        return f"❌ Failed to generate image: {str(e)}"


tools = [
    trace_claim_lineage,
    web_search,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    calculator,
    generate_image
]