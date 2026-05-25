"""Semantic Skills Plugin — pre-load skills at the gateway dispatch level.

This plugin does two things:

1. pre_gateway_dispatch hook (preload.py): Before every incoming message
   reaches the agent, embeds the message against the skill index and
   rewrites the message text to include the top-match skill content.
   The system prompt stays at ~226 tokens — fully cache-friendly.

2. search_skills tool (tools.py): On-demand semantic skill search for
   cases where the pre-loaded skill is insufficient.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("hermes.semantic_skills")

_PLUGIN_DIR = Path(__file__).resolve().parent


def register(ctx) -> None:
    """Register the pre_gateway_dispatch hook and search_skills tool."""

    # 1. Register the pre_gateway_dispatch hook — fires at the gateway
    #    level before auth and agent dispatch. Rewrites user messages
    #    to include pre-loaded skill content.
    from .preload import on_pre_gateway_dispatch

    ctx.register_hook("pre_gateway_dispatch", on_pre_gateway_dispatch)

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
        "Semantic Skills plugin loaded — pre_gateway_dispatch hook + search_skills tool. "
        "Skill pre-loading at gateway level with cache-friendly message injection."
    )
