"""Embedding store — manages the skill embedding index.

Multi-backend architecture with auto-detection and graceful degradation:

  Priority 1: litellm  — LiteLLM proxy (local, zero-latency, already running)
  Priority 2: openai   — OpenAI embeddings API (cloud, needs OPENAI_API_KEY)
  Priority 3: sentence_transformers — local CPU (all-MiniLM-L6-v2, auto-download)
  Priority 4: keyword  — TF-IDF via pure Python (stdlib, zero-dependency fallback)

The store pre-computes embeddings for all SKILL.md files at build time.
Query embeddings are computed on-the-fly using the best available backend.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent / "data"
_INDEX_FILE = _DATA_DIR / "skills_index.json"
_EMBEDDINGS_FILE = _DATA_DIR / "skills_embeddings.npz"
_SKILLS_DIR = Path(os.path.expanduser("~/.hermes/skills"))
_EMBEDDING_DIM = 768  # nomic-embed-text-v2-moe / all-MiniLM-L6-v2

# ── Backend detection ──────────────────────────────────────────────────
_backend_priority = []


def _detect_backends() -> list[str]:
    """Return available embedding backends in priority order."""
    available = []

    # 1. LiteLLM — check if proxy is running
    try:
        import urllib.request

        master_key = os.environ.get("LITELLM_MASTER_KEY", "")
        if not master_key:
            try:
                import yaml
                with open(os.path.expanduser("~/.litellm/proxy_config.yaml")) as f:
                    config = yaml.safe_load(f)
                master_key = config.get("general_settings", {}).get("master_key", "")
            except Exception:
                pass

        req = urllib.request.Request("http://127.0.0.1:4000/health")
        if master_key:
            req.add_header("Authorization", f"Bearer {master_key}")
        urllib.request.urlopen(req, timeout=2)
        available.append("litellm")
    except Exception:
        pass

    # 2. OpenAI — check for API key
    if os.environ.get("OPENAI_API_KEY"):
        available.append("openai")

    # 3. sentence-transformers — check if importable
    try:
        import sentence_transformers  # noqa: F401
        available.append("sentence_transformers")
    except ImportError:
        pass

    # 4. Keyword — always available (stdlib)
    available.append("keyword")

    return available


def _get_backend() -> str:
    """Return the best available backend."""
    backends = _detect_backends()
    if not backends:
        return "keyword"
    return backends[0]


def embed_text(text: str, backend: str | None = None) -> list[float]:
    """Embed a single text using the best available backend.

    Args:
        text: Text to embed
        backend: Force a specific backend ('litellm', 'openai', 'sentence_transformers', 'keyword')

    Returns:
        List of floats (embedding vector)
    """
    if backend is None:
        backend = _get_backend()

    if backend == "litellm":
        return _embed_litellm(text)
    elif backend == "openai":
        return _embed_openai(text)
    elif backend == "sentence_transformers":
        return _embed_sentence_transformers(text)
    else:
        return _embed_keyword(text)


def embed_batch(texts: list[str], backend: str | None = None,
                on_backend_failure: str = "keyword") -> list[list[float]]:
    """Embed multiple texts. Uses batch APIs where available.

    If the primary backend fails, falls back to on_backend_failure."""
    if backend is None:
        backend = _get_backend()

    try:
        if backend == "litellm":
            return [_embed_litellm(t) for t in texts]
        elif backend == "openai":
            return _embed_openai_batch(texts)
        elif backend == "sentence_transformers":
            return _embed_sentence_transformers_batch(texts)
        else:
            return [_embed_keyword(t) for t in texts]
    except Exception as e:
        if backend != on_backend_failure:
            import sys
            print(f"   ⚠️  {backend} backend failed: {e}", file=sys.stderr)
            print(f"   Falling back to {on_backend_failure}", file=sys.stderr)
            if on_backend_failure == "keyword":
                _build_keyword_index(texts)
            return embed_batch(texts, backend=on_backend_failure,
                               on_backend_failure=on_backend_failure)
        raise


# ── Backend implementations ────────────────────────────────────────────

def _embed_litellm(text: str) -> list[float]:
    """Embed via LiteLLM proxy using nomic-embed-text."""
    import urllib.request

    master_key = os.environ.get("LITELLM_MASTER_KEY", "")
    if not master_key:
        try:
            import yaml
            with open(os.path.expanduser("~/.litellm/proxy_config.yaml")) as f:
                config = yaml.safe_load(f)
            master_key = config.get("general_settings", {}).get("master_key", "")
        except Exception:
            pass

    body = json.dumps({
        "model": "text-embedding-nomic-embed-text-v2-moe",
        "input": text,
    }).encode()

    req = urllib.request.Request(
        "http://127.0.0.1:4000/v1/embeddings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {master_key}",
        },
    )
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
    return data["data"][0]["embedding"]


def _embed_openai(text: str) -> list[float]:
    """Embed via OpenAI API."""
    import urllib.request

    api_key = os.environ["OPENAI_API_KEY"]
    body = json.dumps({
        "model": "text-embedding-3-small",
        "input": text,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    emb = data["data"][0]["embedding"]
    # Pad/truncate to standard dim
    if len(emb) < _EMBEDDING_DIM:
        emb = emb + [0.0] * (_EMBEDDING_DIM - len(emb))
    return emb[:_EMBEDDING_DIM]


def _embed_openai_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed via OpenAI."""
    import urllib.request

    api_key = os.environ["OPENAI_API_KEY"]
    body = json.dumps({
        "model": "text-embedding-3-small",
        "input": texts,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())
    results = []
    for item in sorted(data["data"], key=lambda x: x["index"]):
        emb = item["embedding"]
        if len(emb) < _EMBEDDING_DIM:
            emb = emb + [0.0] * (_EMBEDDING_DIM - len(emb))
        results.append(emb[:_EMBEDDING_DIM])
    return results


_SENTENCE_MODEL = None


def _get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SENTENCE_MODEL


def _embed_sentence_transformers(text: str) -> list[float]:
    """Embed via local sentence-transformers (CPU)."""
    model = _get_sentence_model()
    return model.encode(text).tolist()


def _embed_sentence_transformers_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed via local sentence-transformers."""
    model = _get_sentence_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()


# ── TF-IDF Keyword backend (stdlib only) ───────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _compute_idf(corpus: list[list[str]]) -> dict[str, float]:
    """Compute IDF for a corpus."""
    doc_count = len(corpus)
    if doc_count == 0:
        return {}
    df: dict[str, int] = {}
    for doc in corpus:
        for word in set(doc):
            df[word] = df.get(word, 0) + 1
    return {w: math.log((doc_count + 1) / (df[w] + 1)) + 1 for w in df}


def _tfidf_vector(tokens: list[str], idf: dict[str, float],
                  vocab: list[str]) -> list[float]:
    """Create TF-IDF vector for tokens."""
    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    max_tf = max(tf.values()) if tf else 1
    return [((tf.get(w, 0) / max_tf) * idf.get(w, 0)) for w in vocab]


# Global TF-IDF state — built once when using keyword backend
_tfidf_vocab: list[str] = []
_tfidf_idf: dict[str, float] = {}


def _build_keyword_index(texts: list[str]) -> None:
    """Build TF-IDF vocabulary from a corpus."""
    global _tfidf_vocab, _tfidf_idf
    corpus = [_tokenize(t) for t in texts]
    # Collect all unique words
    all_words: set[str] = set()
    for doc in corpus:
        all_words.update(doc)
    _tfidf_vocab = sorted(all_words)
    _tfidf_idf = _compute_idf(corpus)


def _embed_keyword(text: str) -> list[float]:
    """Embed via TF-IDF (stdlib only)."""
    global _tfidf_vocab, _tfidf_idf
    if not _tfidf_vocab:
        # Try to load from disk
        loaded = load_tfidf_vocab()
        if loaded:
            _tfidf_vocab, _tfidf_idf = loaded
        else:
            return [0.0] * _EMBEDDING_DIM
    tokens = _tokenize(text)
    vec = _tfidf_vector(tokens, _tfidf_idf, _tfidf_vocab)
    # Pad/truncate to standard dim
    if len(vec) < _EMBEDDING_DIM:
        vec = vec + [0.0] * (_EMBEDDING_DIM - len(vec))
    return vec[:_EMBEDDING_DIM]


# ── Index management ───────────────────────────────────────────────────

def index_exists() -> bool:
    """Check if embedding index exists on disk."""
    return _INDEX_FILE.exists() and _EMBEDDINGS_FILE.exists()


def load_index() -> tuple[list[dict], list[list[float]]] | None:
    """Load embedding index from disk. Returns (metadata_list, embeddings_list) or None."""
    if not index_exists():
        return None
    try:
        with open(_INDEX_FILE) as f:
            metadata = json.load(f)
    except Exception:
        return None
    try:
        import numpy as np
        data = np.load(_EMBEDDINGS_FILE)
        embeddings = data["embeddings"].tolist()
    except Exception:
        return None
    return metadata, embeddings


def save_index(metadata: list[dict], embeddings: list[list[float]],
               tfidf_vocab: list[str] | None = None,
               tfidf_idf: dict[str, float] | None = None) -> None:
    """Save embedding index to disk."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(_INDEX_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        import numpy as np
        np.savez_compressed(_EMBEDDINGS_FILE, embeddings=np.array(embeddings, dtype=np.float32))
    except ImportError:
        with open(str(_EMBEDDINGS_FILE).replace(".npz", ".json"), "w") as f:
            json.dump(embeddings, f)

    # Also save TF-IDF vocabulary if provided
    if tfidf_vocab:
        _vocab_file = _DATA_DIR / "tfidf_vocab.json"
        with open(_vocab_file, "w") as f:
            json.dump({"vocab": tfidf_vocab, "idf": tfidf_idf or {}}, f)


def load_tfidf_vocab() -> tuple[list[str], dict[str, float]] | None:
    """Load TF-IDF vocabulary from disk."""
    _vocab_file = _DATA_DIR / "tfidf_vocab.json"
    if not _vocab_file.exists():
        return None
    try:
        with open(_vocab_file) as f:
            data = json.load(f)
        return data.get("vocab", []), data.get("idf", {})
    except Exception:
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Skill content reading ──────────────────────────────────────────────

def parse_skill_frontmatter(filepath: Path) -> dict[str, str]:
    """Extract frontmatter fields from a SKILL.md file."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return {}

    # Parse YAML frontmatter
    if not content.startswith("---"):
        return {}

    end = content.find("---", 3)
    if end == -1:
        return {}

    fm_text = content[3:end]
    result: dict[str, str] = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip("\"'")
            result[key] = val

    # Also grab full body text (after frontmatter) for better embedding
    result["_body"] = content[end + 3:].strip()[:2000]

    return result


def iter_skills(skills_dir: Path | None = None) -> list[dict]:
    """Scan all SKILL.md files and return metadata."""
    if skills_dir is None:
        skills_dir = _SKILLS_DIR

    results = []
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        fm = parse_skill_frontmatter(skill_file)
        if not fm.get("name"):
            fm["name"] = skill_file.parent.name

        results.append({
            "name": fm.get("name", ""),
            "description": fm.get("description", ""),
            "category": skill_file.parent.parent.name
                        if skill_file.parent.parent != skills_dir
                        else "general",
            "path": str(skill_file),
            "body_preview": fm.get("_body", ""),
        })

    return results


def build_search_text(skill: dict) -> str:
    """Build a rich text representation for embedding."""
    parts = [
        skill.get("name", ""),
        skill.get("description", ""),
        skill.get("category", ""),
        # Include first 500 chars of body for better matching
        skill.get("body_preview", "")[:500],
    ]
    return " ".join(p for p in parts if p)


def read_skill_content(filepath: str) -> str:
    """Read the full SKILL.md content."""
    try:
        return Path(filepath).read_text(encoding="utf-8")
    except Exception:
        return f"(Could not read skill at {filepath})"
