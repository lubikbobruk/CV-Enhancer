"""Phase 4 end-to-end smoke test.

Flow: synthetic CV + synthetic job ad -> chunk -> RRF rank -> pick the
least-relevant chunks -> enhance_chunks() -> print rewrites and reasons.

Run from the repo root: `python -m tests.smoke_phase4`."""

from pathlib import Path

from src.parsing.chunker import chunk_cv
from src.parsing.job_ad_filter import filter_role_relevant
from src.retrieval.embedding_retriever import DenseRetriever, load_minilm
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.generation.gemini_client import enhance_chunks


_FIXTURES = Path(__file__).parent / "phase4_fixtures"


def main() -> None:
    cv_text = (_FIXTURES / "synthetic_cv.txt").read_text(encoding="utf-8")
    job_ad_text = (_FIXTURES / "synthetic_job_ad.txt").read_text(encoding="utf-8")

    model = load_minilm()
    job_ad = filter_role_relevant(job_ad_text, model=model)

    chunks = chunk_cv(cv_text)
    print(f"Parsed {len(chunks)} chunks.\n")

    tfidf = TfidfRetriever()
    tfidf.fit(chunks)
    dense = DenseRetriever(model)
    dense.fit(chunks)

    k = len(chunks)
    fused = reciprocal_rank_fusion([tfidf.query(job_ad, k), dense.query(job_ad, k)], k=k)

    least_relevant = list(reversed(fused))[:3]
    print("Least-relevant chunks (target for rewriting):")
    for chunk_id, score in least_relevant:
        preview = chunks[chunk_id][:80].replace("\n", " ")
        print(f"  chunk_id={chunk_id} rrf={score:.4f} | {preview}...")
    print()

    selected = [(cid, chunks[cid]) for cid, _ in least_relevant]
    result = enhance_chunks(cv_text, job_ad, selected)

    print("=== Gemini rewrites ===\n")
    for rewrite in result.rewrites:
        print(f"--- chunk_id={rewrite.chunk_id} ---")
        print("ORIGINAL:")
        print(chunks[rewrite.chunk_id])
        print("\nREWRITTEN:")
        print(rewrite.rewritten_text)
        print("\nREASON:")
        print(rewrite.reason)
        print()


if __name__ == "__main__":
    main()
