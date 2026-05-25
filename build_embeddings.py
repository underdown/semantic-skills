"""Build the skill embedding index.

Usage:
    python build_embeddings.py [--backend litellm|openai|sentence_transformers|keyword]
                               [--force]

Pre-computes embeddings for all SKILL.md files in ~/.hermes/skills/ and
saves them to ~/.hermes/plugins/jit-skills/data/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add parent to path so we can import the embedding store
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding_store import (
    build_search_text,
    embed_batch,
    _detect_backends,
    _build_keyword_index,
    _tfidf_vocab,
    _tfidf_idf,
    index_exists,
    iter_skills,
    save_index,
)


def main():
    parser = argparse.ArgumentParser(description="Build JIT skill embedding index")
    parser.add_argument("--backend", choices=["litellm", "openai", "sentence_transformers", "keyword"],
                        help="Force a specific embedding backend")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if index exists")
    args = parser.parse_args()

    if index_exists() and not args.force:
        print("✅ Index already exists. Use --force to rebuild.")
        return

    # Determine backend
    backend = args.backend
    if backend is None:
        backends = _detect_backends()
        if not backends:
            print("❌ No embedding backends available. Install one of:")
            print("   - sentence-transformers: pip install sentence-transformers")
            print("   - openai: set OPENAI_API_KEY")
            print("   - keyword: built-in (fallback)")
            sys.exit(1)
        backend = backends[0]

    print(f"🔍 Using embedding backend: {backend}")

    # Scan skills
    print("📂 Scanning skills...")
    skills = iter_skills()
    print(f"   Found {len(skills)} skills")

    if not skills:
        print("❌ No skills found in ~/.hermes/skills/")
        sys.exit(1)

    # Build search texts
    texts = [build_search_text(s) for s in skills]
    total_chars = sum(len(t) for t in texts)
    print(f"   Total search text: {total_chars:,} chars")

    # Embed
    print(f"🧮 Computing embeddings...")
    t0 = time.time()

    if backend == "keyword":
        _build_keyword_index(texts)
        print("   Built TF-IDF vocabulary")

    embeddings = embed_batch(texts, backend=backend)
    elapsed = time.time() - t0
    print(f"   {len(embeddings)} embeddings in {elapsed:.1f}s ({elapsed / max(len(embeddings), 1) * 1000:.0f}ms each)")

    # Build metadata (strip body_preview — not needed in index)
    metadata = [
        {
            "name": s["name"],
            "description": s["description"],
            "category": s["category"],
            "path": s["path"],
        }
        for s in skills
    ]

    # Save (include TF-IDF vocab if using keyword backend)
    print("💾 Saving index...")
    # Access via module to get the mutated global, not the import-time copy
    import embedding_store as _es
    # Always save TF-IDF vocab as fallback for when primary backend is unavailable
    _build_keyword_index(texts)
    tfidf_vocab = list(_es._tfidf_vocab) if _es._tfidf_vocab else None
    tfidf_idf = dict(_es._tfidf_idf) if _es._tfidf_idf else None
    save_index(metadata, embeddings, tfidf_vocab=tfidf_vocab, tfidf_idf=tfidf_idf)
    print(f"✅ Done. Index saved to data/skills_index.json + data/skills_embeddings.npz")
    print(f"   {len(metadata)} skills indexed using {backend}")


if __name__ == "__main__":
    main()
