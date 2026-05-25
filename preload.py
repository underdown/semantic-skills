"""Semantic skill pre-loading hook — gateway dispatch level.

This hook fires at the gateway layer via ``pre_gateway_dispatch``, before
the message reaches the agent loop. It embeds the incoming message against
the semantic skill index and rewrites the message text to include the
top-match skill content at the beginning.

Key advantages over ``pre_llm_call``:
- Fires earlier — before auth, agent wake-up, and tool schema loading
- Skill content becomes part of the raw user message from the start
- System prompt is never touched — perfect exact-prefix caching
- Works at the gateway level, so it applies to ALL platforms uniformly
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

_PLUGIN_DIR = Path(__file__).resolve().parent
_SCORE_THRESHOLD = 0.30  # Minimum cosine similarity to pre-load a skill
_SESSION_CACHE: Dict[str, str] = {}  # session_key → last injected skill name


def on_pre_gateway_dispatch(
    event: Any = None,
    gateway: Any = None,
    session_store: Any = None,
    **kwargs,
) -> Optional[Dict[str, str]]:
    """Pre-load the best-matching skill by rewriting the incoming message.

    Called by the Hermes gateway for every incoming user message before
    auth, pairing, or agent dispatch.  Returns a dict influencing flow:

        {\"action\": \"rewrite\", \"text\": \"[skill]\\n\\n[msg]\"}  — inject skill
        None                                           — normal dispatch

    Args:
        event: ``MessageEvent`` with ``.text`` (str) and ``.source``.
        gateway: ``GatewayRunner`` instance (unused by this hook).
        session_store: Session store (unused by this hook).
    """
    if event is None:
        return None

    text = getattr(event, "text", "") or ""
    if not text.strip():
        return None

    # Quick early exit: skip very short queries
    if len(text.split()) < 3:
        return None

    # Derive a stable session key from the event source so we can
    # avoid re-injecting the same skill on follow-up turns.
    source = getattr(event, "source", None)
    session_key = (
        getattr(source, "chat_id", None)
        or getattr(source, "user_id", None)
        or "default"
    )

    try:
        # Lazy-import embedding_store — defers NumPy until first use.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "embedding_store", _PLUGIN_DIR / "embedding_store.py"
        )
        store = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(store)

        if not store.index_exists():
            return None

        index_data = store.load_index()
        if index_data is None:
            return None

        metadata, embeddings = index_data

        # Embed the incoming message
        backend = store._get_backend()
        try:
            query_embedding = store.embed_text(text, backend=backend)
        except Exception:
            try:
                query_embedding = store.embed_text(text, backend="keyword")
            except Exception:
                return None

        # Rank by cosine similarity — find the single best match
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

        # Read full skill content
        skill_content = store.read_skill_content(best_meta["path"])
        if not skill_content:
            return None

        # Avoid re-injecting the same skill on follow-up turns
        if _SESSION_CACHE.get(session_key) == best_meta["name"]:
            return None

        _SESSION_CACHE[session_key] = best_meta["name"]

        # Build the rewritten message — skill content goes first,
        # then the original user message on the next line.
        rewritten = (
            f"Relevant skill (pre-loaded, score {best_score:.2f}): {best_meta['name']}\n"
            f"{'─' * 60}\n"
            f"{skill_content}\n"
            f"{'─' * 60}\n\n"
            f"If this skill is insufficient, call search_skills for additional matches.\n\n"
            f"{'─' * 40}\n\n"
            f"{text}"
        )

        return {"action": "rewrite", "text": rewritten}

    except Exception:
        # Graceful degradation — never block message dispatch
        return None
