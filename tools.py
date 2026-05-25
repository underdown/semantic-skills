"""search_skills tool — JIT semantic skill retrieval.

Registered as a Hermes tool. Accepts a natural-language query and returns
the top-N matching skills with full content.

Backends (auto-detected in priority order):
  1. litellm  — LiteLLM proxy at localhost:4000
  2. openai   — OpenAI embeddings API
  3. sentence_transformers — local all-MiniLM-L6-v2
  4. keyword  — TF-IDF fallback (stdlib only)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# These are resolved at call time to avoid import errors
# during plugin loading (embedding_store imports numpy conditionally)
_STORE_PATH = Path(__file__).resolve().parent / "embedding_store.py"


def _get_store():
    """Lazy-import embedding_store."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("embedding_store", _STORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEARCH_SKILLS_SCHEMA = {
    "name": "search_skills",
    "description": (
        "Find relevant skills for any task using semantic search. "
        "Call this BEFORE starting any unfamiliar task — it finds specialized "
        "workflows, known pitfalls, API commands, and proven approaches that "
        "general-purpose tools don't know about.\n\n"
        "Describe what you're trying to do in natural language. The tool uses "
        "embeddings to find the best-matching skills, then returns their full "
        "content with step-by-step instructions.\n\n"
        "Examples:\n"
        '  search_skills("debug LiteLLM proxy connection errors")\n'
        '  search_skills("set up GitHub PR workflow and code review")\n'
        '  search_skills("generate pixel art with NES palette")\n'
        '  search_skills("scrape and monitor competitor pricing pages")\n\n'
        "If no good matches are found, proceed without a skill."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language description of what you're trying to do",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of skills to return (default: 5, max: 10)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


def _handle_search_skills(query: str, limit: int = 5, task_id: str = None) -> str:
    """Search for relevant skills and return their full content."""
    store = _get_store()
    t0 = time.time()

    # Clamp limit
    limit = min(max(1, limit), 10)

    # Load index or build on first use
    if not store.index_exists():
        return json.dumps({
            "error": "skill_index_not_built",
            "message": (
                "No skill embedding index found. Run: "
                "python ~/.hermes/plugins/jit-skills/build_embeddings.py"
            ),
            "hint": "This is a one-time setup. After building, skills are available instantly.",
        })

    index_data = store.load_index()
    if index_data is None:
        return json.dumps({"error": "index_load_failed", "message": "Could not load skill index"})

    metadata, embeddings = index_data

    # Determine backend and embed the query
    backend = store._get_backend()

    try:
        query_embedding = store.embed_text(query, backend=backend)
    except Exception as e:
        # Fall back to keyword on any embedding failure
        try:
            query_embedding = store.embed_text(query, backend="keyword")
            backend = "keyword (fallback)"
        except Exception:
            return json.dumps({
                "error": "embedding_failed",
                "message": f"All embedding backends failed: {e}",
                "hint": "Ensure at least one backend is available (litellm, openai, sentence_transformers, or keyword)",
            })

    # Rank by cosine similarity
    scored = []
    for i, emb in enumerate(embeddings):
        if i >= len(metadata):
            break
        score = store.cosine_similarity(query_embedding, emb)
        scored.append((score, metadata[i]))

    scored.sort(key=lambda x: -x[0])

    # Build results
    results = []
    for score, meta in scored[:limit]:
        if score < 0.05:
            continue  # Skip very low relevance

        content = store.read_skill_content(meta["path"])

        results.append({
            "name": meta["name"],
            "description": meta.get("description", ""),
            "category": meta.get("category", "general"),
            "relevance": round(score, 4),
            "content": content,
        })

    elapsed = time.time() - t0

    if not results:
        return json.dumps({
            "query": query,
            "backend": backend,
            "results": [],
            "message": "No relevant skills found. Proceed without a skill.",
            "search_time_ms": round(elapsed * 1000),
        })

    # Format output — return content inline so agent doesn't need a
    # second skill_view() call
    out = {
        "query": query,
        "backend": backend,
        "results_count": len(results),
        "search_time_ms": round(elapsed * 1000),
        "results": results,
    }
    return json.dumps(out, ensure_ascii=False)


def _check_search_skills_available() -> bool:
    """Always available — even keyword backend works with stdlib."""
    return True
