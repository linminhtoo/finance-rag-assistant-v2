"""
Aim: end-to-end test of Financial RAG system on held-out eval questions.

Use these to benchmark changes to chunking, retrieval, reranking, LLM prompting, etc.
Aim to plot pareto frontiers of accuracy vs latency.
Examples:
- multiple LLM calls per query (query rewriting, HyDE, improved retrieval, etc)
- compare different LLM models - smaller models for certain steps vs larger models

-----------------------------------
Create 10-30 eval Qs:
- Simple factual (“What does X section say about Y?”)
- Multi-hop (“Contrast A vs B across docs”)
- Edge case (“When does the doc not support a claim?”)
- Auto-generate candidates and then prune manually:

# scripts/make_eval.py (sketch)
# Take titles/sections → synthesize Qs using your LLM, then manually review.

-----------------------------------
Score with Ragas (LLM-assisted) or simple string-match baselines.

Track:
- Answer correctness
- Groundedness (uses provided context)
- Context precision (are retrieved chunks relevant?)
- If precision is low → increase rerank top_n, reduce chunk size, add hybrid search (BM25 + vectors). For BM25 you can do: retrieve 10 via BM25 (rank-bm25) + 10 via vectors → union → rerank.

-----------------------------------
Observability -> look into `langfuse` or MLFlow for traces

-----------------------------------
Troubleshooting cheat-sheet
- Bad answers / hallucinations → lower temperature; add “must cite” instruction; add verification pass; improve chunking & rerank.
- Irrelevant context → smaller chunks (e.g., 600-800 tokens), larger overlap (150-200), hybrid retrieval, reranker top-k 50→rerank→top-6.
- Latency → reduce k; switch to bge-small embeddings; switch from cross-encoder to ColBERT-style later; consider API LLM.
- Ollama OOM → use a smaller/quantized model: llama3.2:3b-instruct or mistral:7b q4_K_M.
"""

# Backwards-compatible alias: the actual runner lives in scripts/run_eval.py.

"""
Added an end-to-end eval framework with a JSONL dataset format (repos/z_scratch/financial-rag/src/finrag/eval/schema.py (line 9)) plus loaders/savers (repos/z_scratch/financial-rag/src/finrag/eval/io.py (line 1)) and metrics/runner/report (repos/z_scratch/financial-rag/src/finrag/eval/runner.py (line 88), repos/z_scratch/financial-rag/src/finrag/eval/report.py (line 1)).
Implemented SEC HTML chunking + corpus loader so evals work directly on the downloaded EDGAR files (repos/z_scratch/financial-rag/src/finrag/sec_chunking.py (line 20), repos/z_scratch/financial-rag/src/finrag/eval/sec_corpus.py (line 1)).
Implemented synthetic query generation (quant/qual/mixed + optional multi-doc net-income QoQ series for 10-Q corpora) with evidence snippets for human verification (repos/z_scratch/financial-rag/src/finrag/eval/generation.py (line 203), repos/z_scratch/financial-rag/scripts/make_eval_set.py (line 14)).
Implemented an eval runner that indexes a selected subset of filings, runs hybrid retrieval + optional reranking, and outputs *.json + a standalone *.html report (repos/z_scratch/financial-rag/scripts/run_eval.py (line 17)). scripts/bulk_test.py now aliases to this runner (repos/z_scratch/financial-rag/scripts/bulk_test.py (line 40)).
Made retrieval robust to embedding dimension mismatches (e.g. OpenAI vs Mistral vs local embeddings) by inferring vector size at index-time, and added a NoopReranker for offline/no-download runs (repos/z_scratch/financial-rag/src/finrag/retriever.py (line 17)); added fastembed as an embedding-only provider for retrieval-only evals (repos/z_scratch/financial-rag/src/finrag/llm_clients.py (line 91)).
Run it:

Generate eval set: python3 scripts/make_eval_set.py --data-dir ./data --out ./eval/eval_set.jsonl --max-docs 200 --n-quant 30 --n-qual 30 --n-mixed 10 --n-series 5 (repos/z_scratch/financial-rag/README.md (line 29))
Run eval: python3 scripts/run_eval.py --data-dir ./data --eval-set ./eval/eval_set.jsonl --out-dir ./results --max-docs 200 --no-reranker (repos/z_scratch/financial-rag/README.md (line 48))
"""
from run_eval import main


if __name__ == "__main__":
    main()
