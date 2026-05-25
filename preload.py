"""Semantic skill pre-loading hook for Hermes Agent.

This hook fires before every LLM call. It embeds the user's message
against the semantic skill index and injects the top-match skill content
into the user message as ephemeral context.

Key constraint: The skill content is injected into the USER MESSAGE (via
pre_llm_call's context injection), NOT the system prompt. This keeps the
system prompt fully stable and cache-friendly — exactly the pattern that
maximizes exact-prefix prompt caching across turns.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

_PLUGIN_DIR = Path(__file__).resolve().parent
_SCORE_THRESHOLD = 0.30  # Minimum cosine similarity to pre-load a skill
_SESSION_CACHE: dict[str, str] = {}  # session_id → last injected skill name


def on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    is_first_turn: bool = False,
    **kwargs,
) -> Optional[dict]:
    """Pre-load the best-matching skill into the user message.

    On the first turn of a session (or when the user's query changes
    meaningfully), embed the query against the skill index, find the top
    match, and inject its content as ephemeral context.

    Returns:
        dict with "context" key containing the skill content, or None
        if no good match found or backend unavailable.
    """
    if not user_message.strip():
        return None

    # Quick early exit: skip short queries that can't meaningfully embed
    if len(user_message.split()) < 3:
        return None

    try:
        # Lazy-import embedding_store to avoid loading numpy on every
        # Hermes startup — it only loads when the hook first fires.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "embedding_store", _PLUGIN_DIR / "embedding_store.py"
        )
        store = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(store)

        # Load index or bail
        if not store.index_exists():
            return None

        index_data = store.load_index()
        if index_data is None:
            return None

        metadata, embeddings = index_data

        # Embed the query
        backend = store._get_backend()
        try:
            query_embedding = store.embed_text(user_message, backend=backend)
        except Exception:
            try:
                query_embedding = store.embed_text(user_message, backend="keyword")
            except Exception:
                return None

        # Rank by cosine similarity
        best_score = -1.0
        best_meta = None

        for i, emb in enumerate(embeddings):
            if i >= len(metadata):
                break
            score = store.cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_meta = metadata[i]

        if best_score < _SCORE_THRESHOLD or best_meta is None:
            return None

        # Read the full skill content
        skill_content = store.read_skill_content(best_meta["path"])
        if not skill_content:
            return None

        # Avoid re-injecting the same skill on follow-up turns
        if session_id and _SESSION_CACHE.get(session_id) == best_meta["name"]:
            return None

        if session_id:
            _SESSION_CACHE[session_id] = best_meta["name"]

        # Build the context block
        context = (
            f"Relevant skill (pre-loaded, score {best_score:.2f}): {best_meta['name']}\n"
            f"{'─' * 60}\n"
            f"{skill_content}\n"
            f"{'─' * 60}\n\n"
            f"If this skill is insufficient, call search_skills for additional matches.\n\n"
            f"{'─' * 40}\n\n"
        )

        return {"context": context}

    except Exception:
        # Graceful degradation — never block the LLM call
        return None
