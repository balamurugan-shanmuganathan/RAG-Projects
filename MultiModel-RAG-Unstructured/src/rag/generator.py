import json
from langchain_core.messages import HumanMessage
from ..utils.config import gemini_llm, ollama_llm


def get_llm(model_name: str):
    """Return the appropriate LLM based on user selection."""
    return ollama_llm if model_name == "ollama" else gemini_llm


# ── Streaming Answer ──────────────────────────────
def stream_final_answer(chunks, query: str, model_name: str = "gemini"):
    """Stream the RAG answer token-by-token."""
    llm = get_llm(model_name)
    try:
        prompt = _build_prompt(chunks, query)
        message = HumanMessage(content=prompt)

        for chunk in llm.stream([message]):
            content = chunk.content
            if isinstance(content, list):
                yield "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            else:
                yield str(content)

    except Exception as e:
        yield f"\n\n❌ Streaming error ({model_name}): {str(e)}"


# ── Non-streaming Answer ──────────────────────────
def generate_final_answer(chunks, query: str, model_name: str = "gemini") -> str:
    """Generate a complete answer (non-streaming)."""
    llm = get_llm(model_name)
    try:
        prompt = _build_prompt(chunks, query)
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        content = response.content
        if isinstance(content, list):
            return "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        return str(content)
    except Exception as e:
        return f"❌ Generation error ({model_name}): {str(e)}"


# ── Sample Question Generation ────────────────────
def generate_sample_questions(
    chunks, filename: str, model_name: str = "gemini"
) -> list[str]:
    """
    Given retrieved document chunks, ask the LLM to produce 10 diverse
    questions a user might ask about the document.
    Returns a list of question strings.
    """
    llm = get_llm(model_name)

    # Build a compact context (first 4000 chars)
    context_parts = []
    char_budget = 4000
    for chunk in chunks:
        raw = ""
        if "original_content" in chunk.metadata:
            try:
                orig = json.loads(chunk.metadata["original_content"])
                raw = orig.get("raw_text", chunk.page_content)
            except Exception:
                raw = chunk.page_content
        else:
            raw = chunk.page_content
        if char_budget <= 0:
            break
        snippet = raw[:char_budget]
        context_parts.append(snippet)
        char_budget -= len(snippet)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a research assistant. Based on the document content below (from "{filename}"),
generate exactly 10 insightful, diverse questions that a user would naturally ask while analyzing this document.

Rules:
- Questions must be specific to the content, not generic.
- Cover a range of topics: definitions, methods, results, comparisons, implications.
- Each question should be a single clear sentence ending with "?".
- Return ONLY a JSON array of 10 strings with no extra text or markdown fences.

DOCUMENT CONTENT:
{context}

Return format example:
["Question 1?", "Question 2?", ..., "Question 10?"]"""

    try:
        message = HumanMessage(content=[{"type": "text", "text": prompt}])
        response = llm.invoke([message])
        content = response.content
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        content = content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        questions = json.loads(content)
        if isinstance(questions, list):
            return [str(q).strip() for q in questions[:10] if str(q).strip()]
        return []
    except Exception as e:
        print(f"⚠️  Question generation failed: {e}")
        # Return sensible fallbacks
        return [
            f"What is the main topic of {filename}?",
            "What are the key findings discussed in this document?",
            "What methodology is used in this research?",
            "What conclusions does the document reach?",
            "What data sources or datasets are referenced?",
            "What are the limitations mentioned in the document?",
            "How does this work compare to prior research?",
            "What future directions are suggested?",
            "What technical terms are defined in this document?",
            "What practical applications are discussed?",
        ]


# ── Internal Prompt Builder ───────────────────────
def _build_prompt(chunks, query: str) -> list:
    """Build a multimodal message content list."""
    text_parts = [
        f"Based on the following documents, answer this question: {query}\n\n"
        "Citation rule: Use [1], [2], etc. to cite your sources.\n\n"
        "DOCUMENTS:\n"
    ]

    image_parts = []

    for i, chunk in enumerate(chunks):
        text_parts.append(f"--- Document {i+1} ---\n")
        if "original_content" in chunk.metadata:
            try:
                orig = json.loads(chunk.metadata["original_content"])
                raw = orig.get("raw_text", "")
                if raw:
                    text_parts.append(f"TEXT:\n{raw}\n\n")
                for j, table in enumerate(orig.get("tables_html", [])):
                    text_parts.append(f"Table {j+1}:\n{table}\n\n")
                for b64 in orig.get("images_base64", []):
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
            except Exception:
                text_parts.append(f"{chunk.page_content}\n\n")
        else:
            text_parts.append(f"{chunk.page_content}\n\n")

    text_parts.append(
        "\nProvide a clear, comprehensive answer. "
        "If the documents lack sufficient information, state that explicitly.\n\nANSWER:"
    )

    content = [{"type": "text", "text": "".join(text_parts)}]
    content.extend(image_parts)
    return content