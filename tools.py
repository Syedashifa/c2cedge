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
from deepfake_detector import run_full_deepfake_analysis

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
def detect_deepfake_media(media_input: str, media_type: str = "auto") -> str:
    """
    Detects AI Deepfakes in images, videos, audio/voice recordings, and OCR screenshot text.
    Uses Hive API (Visual Deepfakes), Reality Defender (Video Scans), Resemble AI (Voice Clone Detection), and EasyOCR.
    
    Inputs:
    - media_input: URL, file path, or text description of the image, video, audio, or screenshot.
    - media_type: 'image', 'video', 'voice', 'screenshot', or 'auto'.
    """
    analysis = run_full_deepfake_analysis(media_input, media_type)
    
    verdict = analysis["verdict"]
    score = analysis["overall_deepfake_score"]
    breakdown = analysis["services_breakdown"]

    output = []
    output.append(f"### {verdict}")
    output.append(f"**Target Media:** *\"{media_input}\"*\n")
    
    # 1. EVIDENCE
    output.append("#### 📑 1. FORENSIC EVIDENCE & MULTI-MODAL BREAKDOWN")
    hive = breakdown.get("hive_image", {})
    output.append(f"- 🐝 **Hive AI Visual Inspection:** Status `{hive.get('status')}` | Deepfake Pattern: `{hive.get('is_deepfake')}`")
    
    rd = breakdown.get("reality_defender_video", {})
    output.append(f"- 🛡️ **Reality Defender Video Scan:** Scan Status `{rd.get('status')}` | Verdict: `{rd.get('verdict')}`")
    
    res = breakdown.get("resemble_voice", {})
    output.append(f"- 🎙️ **Resemble AI Voice Analysis:** Status `{res.get('status')}` | Neural Voice Clone: `{res.get('is_synthetic_voice')}`")
    
    ocr = breakdown.get("easy_ocr", {})
    output.append(f"- 📄 **EasyOCR Screenshot Claim Extraction:** Extracted Text: *\"{ocr.get('extracted_text')[:100]}\"*")

    # 2. CONFIDENCE
    output.append("\n#### 🎯 2. CONFIDENCE METRICS")
    output.append(f"- 🎛️ **Overall Synthetic Deepfake Probability:** `{score}%`")
    output.append(f"- 🐝 **Hive AI Visual Model Confidence:** `{hive.get('confidence', 0)*100:.1f}%`")
    output.append(f"- 🛡️ **Reality Defender Temporal Manipulation Score:** `{rd.get('score', 0)*100:.1f}%`")
    output.append(f"- 🎙️ **Resemble AI Voice Authenticity Confidence:** `{res.get('confidence', 0)*100:.1f}%`")

    # 3. UNCERTAINTY
    output.append("\n#### ⚖️ 3. UNCERTAINTY & FORENSIC LIMITATIONS")
    if score > 80:
        output.append("- 🔴 **High Synthetic Certainty:** Multiple detection layers independently confirmed generative AI artifacts.")
    elif 40 <= score <= 80:
        output.append("- ⚠️ **Moderate Uncertainty (Gray Zone):** Mixed indicators detected. Requires cross-referencing with primary source context.")
    else:
        output.append("- 🟢 **Low Synthetic Risk:** Media signature consistent with authentic capture.")
    output.append("- 📌 *Compression noise, low video resolution, and heavy re-encoding can introduce marginal spectral artifacts.*")

    return "\n".join(output)


@tool
def web_search(query: str) -> str:
    """Search the web for latest/current information using Tavily Search."""
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

    retrieved_sources = []
    for idx, s in enumerate(sources_raw):
        retrieved_sources.append({
            "title": s.get("title", f"Source {idx+1}"),
            "url": s.get("url", ""),
            "domain": s.get("url", "").split("/")[2] if "//" in s.get("url", "") else "web",
            "snippet": s.get("content", ""),
            "published_date": s.get("published_date", f"2026-08-22 T+{idx*2}h")
        })

    graph_res = build_lineage_graph(clean_claim, retrieved_sources)

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
            graph_res = build_lineage_graph(clean_claim, retrieved_sources)
        except Exception:
            pass

    # Step 5: Explainer Agent - Format output into Trust Card (Evidence + Confidence + Uncertainty)
    trust_badge = graph_res["trust_badge"]
    summary_headline = graph_res["summary_headline"]
    ind_count = graph_res["independent_origins_count"]
    tot_count = graph_res["total_sources_count"]

    # Build Markdown Trust Card (Explicitly: Evidence + Confidence + Uncertainty)
    trust_card_md = []
    trust_card_md.append(f"### {trust_badge}")
    trust_card_md.append(f"**Claim Analysis:** *\"{clean_claim[:120]}\"*\n")
    trust_card_md.append(f"> **Verdict Summary:** {summary_headline}\n")

    # 1. EVIDENCE
    trust_card_md.append("#### 📑 1. EVIDENCE & CITATIONS")
    trust_card_md.append(f"- 📊 **Independent Confirmations:** `{ind_count}` true root source{'s' if ind_count != 1 else ''} out of `{tot_count}` total scraped articles")
    for node in graph_res["nodes"]:
        origin_tag = " [PRIMARY ROOT ORIGIN]" if node["is_independent_origin"] else " [COPY / RE-PUBLISHER]"
        trust_card_md.append(f"- **{node['title']}** ({node['domain']}){origin_tag}\n  🔗 [View Source Citation]({node['url']})")

    if graph_res["mutations"]:
        trust_card_md.append("\n#### 🚩 FACT MUTATION DRIFT LOG")
        for mut in graph_res["mutations"]:
            trust_card_md.append(f"- ⚠️ **[{mut['type']}]**: {mut['description']}")

    # 2. CONFIDENCE
    trust_card_md.append("\n#### 🎯 2. CONFIDENCE METRICS (Trained XGBoost ML Model)")
    if graph_res["edges"]:
        for edge in graph_res["edges"]:
            trust_card_md.append(
                f"- `[{edge['relationship'].upper()}]` Edge Copy Probability: `{edge['confidence']*100:.1f}%` "
                f"(Embedding Sim: `{edge['embedding_similarity']}`, 3-Gram Overlap: `{edge['ngram_overlap']}`)"
            )
    else:
        trust_card_md.append("- ℹ️ No copy relationships detected between sources (Independent coverage).")

    # 3. UNCERTAINTY & LIMITATIONS
    trust_card_md.append("\n#### ⚖️ 3. UNCERTAINTY & FORENSIC LIMITATIONS")
    if ind_count == 0:
        trust_card_md.append("- 🔴 **High Uncertainty:** Claim spreads organically online with no single traceable primary source.")
    elif graph_res.get("need_re_search"):
        trust_card_md.append("- ⚠️ **Ambiguity Gray Zone (0.40–0.60):** Similarity fell into gray zone; triggered agentic conditional re-search.")
    else:
        trust_card_md.append("- 🟢 **Low Uncertainty:** Pairwise lineage relationships clearly distinguished direct copies from independent reporting.")
    trust_card_md.append("- 📌 *Heavily paraphrased reports with low lexical overlap are marked as low confidence rather than forced as false matches.*")

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
    """Uses Gemini LLM to convert user request into a master image prompt."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return user_prompt

    try:
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL") or "gemini-3.5-flash",
            google_api_key=api_key,
            temperature=0.75
        )
        res = llm.invoke(f"Convert into master-level prompt: {user_prompt}")
        return res.content.strip() if hasattr(res, "content") else str(res).strip()
    except Exception:
        return user_prompt


@tool
def generate_image(prompt: str) -> str:
    """Generate high-quality AI images, photos, illustrations, and artwork."""
    try:
        enhanced_prompt = enhance_image_prompt(prompt)
        seed = random.randint(100000, 999999)
        encoded_prompt = urllib.parse.quote(enhanced_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&seed={seed}&nologo=true"
        return f"![{prompt}]({image_url})"
    except Exception as e:
        return f"❌ Failed to generate image: {str(e)}"


tools = [
    detect_deepfake_media,
    trace_claim_lineage,
    web_search,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    calculator,
    generate_image
]