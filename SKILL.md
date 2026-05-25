---
name: semantic-skills
description: Semantic skill pre-loading for Hermes Agent. Embeds user queries against a skill index and injects the top-match skill content into the user message via pre_llm_call hook. Keeps the system prompt at ~226 tokens — fully cache-friendly.
---

# Semantic Skills

This plugin provides semantic skill pre-loading for Hermes Agent. Before each LLM call, the user's query is embedded against a pre-built skill index. The top-match skill is injected into the user message as ephemeral context — no tool call needed.

## Architecture

```
User message → pre_llm_call hook → embed query → cosine similarity over skill vectors
                                              → inject top-match SKILL.md into user message
                                              → system prompt stays at 226 tokens (unchanged)
```

## Key Design Decisions

- **Skill content injected into user message, NOT system prompt.** This keeps the system prompt byte-identical across turns, maximizing exact-prefix prompt caching.
- **Score threshold: 0.30.** Below this, the hook passes through without injection. The agent can still call `search_skills` manually.
- **Session cache:** On follow-up turns in the same session, the same skill is not re-injected.
- **Graceful degradation:** If embedding fails or the index is missing, the hook returns None and the LLM call proceeds normally.

## Related Files

- `preload.py` — pre_llm_call hook implementation
- `tools.py` — search_skills tool (fallback)
- `embedding_store.py` — 4 embedding backends + index management
- `build_embeddings.py` — one-time index builder
