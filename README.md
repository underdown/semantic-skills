# semantic-skills

Semantic skill pre-loading for Hermes Agent. Embeds user queries against a skill index and injects the top-match skill as ephemeral context — with the system prompt staying at ~226 tokens regardless of how many skills you install.

**Cache-friendly by design.** The skill content is injected into the user message via `pre_llm_call`, not into the system prompt. The system prompt remains byte-identical across turns — maximizing exact-prefix prompt caching on every provider.

## What changed from jit-skills

`semantic-skills` is the successor to `jit-skills`. The core change:

| Feature | jit-skills (v1) | semantic-skills (v2) |
|---------|----------------|---------------------|
| Skill delivery | Agent calls search_skills tool → tool response | Pre-loaded via pre_gateway_dispatch hook → message rewritten |
| search_skills tool | Required for every task | Downgraded to fallback (additional lookups) |
| First-turn tool call | Always (1,500-3,000 tokens) | Eliminated |
| System prompt | 226 tokens, cache-friendly | 226 tokens, cache-friendly (unchanged) |
| Injection point | Tool result in messages array | Gateway-level message rewrite (before agent dispatch) |

## How it works

```
User sends: "debug the litellm proxy timeout"
       ↓
pre_llm_call hook fires
       ↓
Embed user query against 476+ skill vectors
       ↓
Top match: litellm-proxy-debug (score 0.68)
       ↓
Inject full SKILL.md content into user message
       ↓
LLM sees: [system prompt: 226 tokens] [user msg: skill + query]
       ↓
No tool call needed — skill is already in context
```

## Token savings

| Mode | System prompt tokens | First-turn tool call |
|------|---------------------|---------------------|
| `full` (original) | ~4,200 (all skills listed) | No |
| `jit` (old) | ~226 | Yes (1,500-3,000 tokens) |
| `semantic` (v2) | ~226 | **No** |

Total per-session savings vs full mode: ~4,000 system prompt tokens + 1,500-3,000 tool-call tokens = **5,500-7,000 tokens saved per session**.

## Installation

```bash
# Install the plugin
hermes plugins install underdown/semantic-skills --enable

# Rebuild the embedding index (required after install)
cd ~/.hermes/plugins/semantic-skills
python3 build_embeddings.py --backend litellm --force

# Enable semantic mode
hermes config set skills.mode semantic

# Restart the gateway
hermes gateway restart
```

## Config

```yaml
skills:
  mode: semantic   # "semantic" for pre-loaded skills, "full" for original, "jit" for v1 behavior
```

## Embedding backends

Auto-detected in priority order. Falls back gracefully.

| # | Backend | Model | Latency | Requires |
|---|---------|-------|---------|---------|
| 1 | `litellm` | nomic-embed-text-v2-moe | ~150ms | LM Studio on local GPU |
| 2 | `openai` | text-embedding-3-small | ~200ms | OPENAI_API_KEY |
| 3 | `sentence_transformers` | all-MiniLM-L6-v2 (CPU) | ~50ms | pip install sentence-transformers |
| 4 | `keyword` | TF-IDF (stdlib only) | <1ms | Nothing |

## Score threshold

The pre-load hook only injects skills with cosine similarity >= 0.30 (configurable in `preload.py`). Below that threshold, the hook passes through without injection and the agent can still call `search_skills` manually.

## From jit-skills

If you're migrating from `hermes-jit-skills`:

1. Uninstall the old plugin: `hermes plugins uninstall jit-skills`
2. Install this one: `hermes plugins install underdown/semantic-skills --enable`
3. Run the re-apply script to update source patches: `bash /root/reapply-jit-patches.sh`
4. Set the config: `hermes config set skills.mode semantic`
5. Restart: `hermes gateway restart`

## License

MIT
