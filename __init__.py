"""Semantic Skills Plugin — register the pre-load hook + search_skills tool.

This plugin does two things:

1. pre_llm_call hook (preload.py): Before each LLM call, embeds the
   user's query against the skill index and injects the top-match skill
   content into the user message. The system prompt stays at ~226 tokens —
   fully cache-friendly across turns.

2. search_skills tool (tools.py): On-demand semantic skill search for
   cases where the pre-loaded skill is insufficient or the agent needs
   additional matches.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("hermes.semantic_skills")

_PLUGIN_DIR = Path(__file__).resolve().parent


def register(ctx) -> None:
    """Register the pre-load hook and search_skills tool."""

    # 1. Register the pre_llm_call hook — fires before every LLM call
    from .preload import on_pre_llm_call

    ctx.register_hook("pre_llm_call", on_pre_llm_call)

    # 2. Register the search_skills tool — fallback for additional lookups
    from .tools import (
        SEARCH_SKILLS_SCHEMA,
        _check_search_skills_available,
        _handle_search_skills,
    )

    ctx.register_tool(
        name="search_skills",
        toolset="skills",
        schema=SEARCH_SKILLS_SCHEMA,
        handler=_handle_search_skills,
        check_fn=_check_search_skills_available,
        emoji="🔎",
    )

    logger.info(
        "Semantic Skills plugin loaded — pre_llm_call hook + search_skills tool. "
        "Skill pre-loading with cache-friendly message injection."
    )
