#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from finrag.eval.io import load_jsonl
from finrag.eval.judges import JudgeSpec, get_judge_client, get_judge_spec, run_judge
from finrag.eval.runner import save_json
from finrag.eval.scoring import build_context
from finrag.eval.schema import EvalGeneration, EvalQuery

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_labels(csv_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            qid = (row.get("query_id") or row.get("id") or "").strip()
            if not qid:
                continue
            raw = (row.get("human_label") or row.get("label") or "").strip()
            if raw == "":
                continue
            try:
                lab = int(raw)
            except ValueError:
                continue
            if lab not in (0, 1):
                continue
            out[qid] = lab
    return out


def _spec_from_id(judge_id: str) -> JudgeSpec:
    try:
        return get_judge_spec(judge_id)
    except ValueError as e:
        raise SystemExit(str(e)) from e


def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # Positive class is FAIL (1): we care about catching defects.
        "precision_fail": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fail": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fail": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Align an LLM-as-a-judge against human labels for a specific run.")
    ap.add_argument("--run-dir", required=True, help="Run directory containing eval_queries.jsonl + generations.jsonl.")
    ap.add_argument(
        "--labels-csv",
        default=None,
        help="CSV with human labels (defaults to <run-dir>/review.csv). Uses `human_label` (0=pass,1=fail).",
    )
    ap.add_argument(
        "--judge",
        default="faithfulness_v1",
        choices=["faithfulness_v1", "factual_correctness_v1", "refusal_v1", "focus_v1", "comparison_v1"],
        help="Which judge prompt to evaluate.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-fraction", type=float, default=0.75)
    ap.add_argument(
        "--eval-test",
        action="store_true",
        help="Also evaluate the held-out test split (use only after you're done tuning the judge prompt).",
    )
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-context-chars", type=int, default=14_000)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    eval_queries_path = run_dir / "eval_queries.jsonl"
    generations_path = run_dir / "generations.jsonl"
    labels_path = Path(args.labels_csv).expanduser().resolve() if args.labels_csv else (run_dir / "review.csv")

    if not eval_queries_path.exists():
        raise SystemExit(f"Missing: {eval_queries_path}")
    if not generations_path.exists():
        raise SystemExit(f"Missing: {generations_path}")
    if not labels_path.exists():
        raise SystemExit(f"Missing: {labels_path}")

    labels = _load_labels(labels_path)
    if not labels:
        raise SystemExit("No human labels found in labels CSV (fill `human_label` with 0/1).")

    queries = load_jsonl(eval_queries_path, EvalQuery)
    gens = load_jsonl(generations_path, EvalGeneration)
    q_by_id = {q.id: q for q in queries}
    g_by_id = {g.query_id: g for g in gens}

    spec = _spec_from_id(args.judge)
    llm = get_judge_client(provider=args.judge_provider, chat_model=args.judge_model, base_url=args.judge_base_url)

    examples: list[tuple[str, int]] = []
    for qid, lab in labels.items():
        q = q_by_id.get(qid)
        g = g_by_id.get(qid)
        if q is None or g is None or g.error:
            continue
        examples.append((qid, lab))
    # Make the split reproducible even if the CSV row order changes.
    examples.sort(key=lambda t: t[0])
    if args.max_items is not None:
        examples = examples[: max(0, int(args.max_items))]
    if not examples:
        raise SystemExit("No usable labeled examples (need matching generations without errors).")

    ids = [qid for qid, _ in examples]
    y = [lab for _, lab in examples]

    # Dev/test split (stratified when possible).
    try:
        dev_ids, test_ids, dev_y, test_y = train_test_split(
            ids, y, train_size=float(args.dev_fraction), random_state=int(args.seed), stratify=y
        )
    except Exception:
        dev_ids, test_ids, dev_y, test_y = train_test_split(
            ids, y, train_size=float(args.dev_fraction), random_state=int(args.seed), stratify=None
        )

    def _predict(split_ids: list[str]) -> tuple[list[int], list[int], list[dict]]:
        y_true: list[int] = []
        y_pred: list[int] = []
        rows: list[dict] = []
        for qid in split_ids:
            q = q_by_id[qid]
            g = g_by_id[qid]
            ctx = build_context(list(g.top_chunks or []), max_chars=args.judge_context_chars)
            expected = None
            evidence = None
            if spec.judge_id == "factual_correctness_v1" and q.factual is not None:
                n = q.factual.expected_numeric
                expected = f"value={n.value}, scale={n.scale or ''}, unit={n.unit}"
                evidence = q.factual.golden_evidence.snippet

            out, raw = run_judge(
                llm,
                spec,
                question=q.question,
                answer=(g.final_answer or ""),
                context=ctx,
                expected=expected,
                evidence=evidence,
            )
            y_true.append(int(labels[qid]))
            y_pred.append(int(out.prediction))
            rows.append(
                {
                    "query_id": qid,
                    "label": int(labels[qid]),
                    "prediction": int(out.prediction),
                    "explanation": out.explanation_sketchpad,
                    "raw": raw,
                }
            )
        return y_true, y_pred, rows

    dev_true, dev_pred, dev_rows = _predict(list(dev_ids))
    test_true: list[int] | None = None
    test_pred: list[int] | None = None
    test_rows: list[dict] | None = None
    if args.eval_test and test_ids:
        test_true, test_pred, test_rows = _predict(list(test_ids))

    report = {
        "judge_id": spec.judge_id,
        "labels_csv": str(labels_path),
        "dev_fraction": float(args.dev_fraction),
        "seed": int(args.seed),
        "split": {"dev_n": len(dev_ids), "test_n": len(test_ids)},
        "test_evaluated": bool(args.eval_test),
        "dev": _metrics(dev_true, dev_pred),
    }
    if test_true is not None and test_pred is not None:
        report["test"] = _metrics(test_true, test_pred)

    stamp = _timestamp()
    out_json = run_dir / f"judge_alignment.{spec.judge_id}.{stamp}.json"
    out_dev = run_dir / f"judge_alignment.{spec.judge_id}.{stamp}.dev.jsonl"
    out_test = run_dir / f"judge_alignment.{spec.judge_id}.{stamp}.test.jsonl"

    save_json(report, out_json)
    out_dev.write_text("\n".join([json.dumps(r, ensure_ascii=False) for r in dev_rows]) + "\n", encoding="utf-8")
    if test_rows is not None:
        out_test.write_text("\n".join([json.dumps(r, ensure_ascii=False) for r in test_rows]) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_dev}")
    print(f"Dev metrics: {report['dev']}")
    if test_rows is not None and "test" in report:
        print(f"Wrote: {out_test}")
        print(f"Test metrics: {report['test']}")
    else:
        print("Test split held out (no test metrics reported). Re-run with `--eval-test` when ready.")


if __name__ == "__main__":
    main()
