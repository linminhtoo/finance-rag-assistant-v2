# finance-rag-assistant-v2

RAG experiments over SEC filings (e.g. 10-K / 10-Q) with an end-to-end evaluation workflow.

## Configuration (.env)

Put secrets and local configuration in a project root `.env` file (gitignored). Start from:

```bash
cp .env.example .env
```

## Data: download SEC filings

`scripts/download.py` downloads EDGAR filings into a local folder:

- `data/10k_raw/*.html` (raw primary documents)
- `data/meta/*.json` (ticker, CIK, filing date, accession, source URL, etc.)

Example:

```bash
python3 scripts/download.py
```

## Evaluation: synthetic eval set + runner

This repo includes an eval framework designed for iterative RAG improvements:

- Quantitative questions: “net income”, “revenue”, “R&D”, etc. (regex-extracted “silver” answers + evidence snippets)
- Qualitative questions: investing angles like R&D priorities, long-term vision, uncertainties, and competitive dynamics
- Mixed questions: combine a numeric value with qualitative drivers

### 1) Generate an eval set (JSONL)

```bash
python3 scripts/make_eval_set.py \
  --data-dir ./data \
  --out ./eval/eval_set.jsonl \
  --max-docs 200 \
  --n-quant 30 --n-qual 30 --n-mixed 10 --n-series 5
```

Each JSONL line is a `finrag.eval.schema.EvalItem` with:
- `question` (synthetic, human-reviewable)
- `expected_numeric` and/or `expected_key_points` (optional “silver” ground truth)
- `evidences` including `doc_id`, `chunk_id`, and a text `snippet`
- `verification.status` (`unverified` by default) for human QA

### 2) Run the eval

This will chunk + index the selected SEC filings, run retrieval (and optionally answer generation),
and emit a JSON run file + a lightweight HTML report.

```bash
python3 scripts/run_eval.py \
  --data-dir ./data \
  --eval-set ./eval/eval_set.jsonl \
  --out-dir ./results \
  --max-docs 200 \
  --no-reranker
```

To include answer generation/scoring, pass `--do-answer` and configure an LLM provider:

- `LLM_PROVIDER=mistral` (requires `MISTRAL_API_KEY`)
- `LLM_PROVIDER=openai` (requires `OPENAI_API_KEY`)

For retrieval-only runs without chat, you can use local embeddings:

```bash
python3 scripts/run_eval.py ... --llm-provider fastembed --no-reranker
```

Notes:
- `CrossEncoderReranker` defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2` and may need model files available locally; use `--no-reranker` if you can’t download.
- The quantitative/qualitative “ground truth” is intentionally “silver” and meant to be verified/edited by a human before treating metrics as definitive.

## Ingestion: HTML -> Markdown (Marker + OCR LLM)

`scripts/process_html_to_markdown.py` converts SEC filing HTML to PDFs and then to markdown via Marker (optionally using an OpenAI-compatible multimodal LLM).

To estimate prompt size with a HuggingFace tokenizer/processor, enable:
- `--log-prompt-token-count` to log best-effort token counts per request
- `--hf-model-id <hf-id>` to override which HF tokenizer/processor is used (defaults to `--openai-model`)
- `--max-prompt-tokens N` to skip calls that exceed `N` tokens (best-effort)
