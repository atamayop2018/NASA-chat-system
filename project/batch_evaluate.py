#!/usr/bin/env python3
"""
Batch evaluation runner for the NASA RAG system.

Reads a flat-text evaluation dataset (default: ``evaluation_dataset.txt``),
runs every question end-to-end through the RAG pipeline (retrieval -> LLM),
then computes RAGAS metrics over the full batch in a single ``evaluate()``
call and prints per-question + aggregate scores.

Usage:
    python batch_evaluate.py
    python batch_evaluate.py --dataset evaluation_dataset.txt --k 3 \
        --report-json batch_report.json --report-csv batch_report.csv
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

# Load environment (OPENAI_API_KEY, OPENAI_BASE_URL for the Vocareum proxy, ...)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import rag_client
import llm_client
import ragas_evaluator


QUESTION_HEADER_RE = re.compile(r"^Q(\d+)\.\s*(.+)$")
EXPECTED_HEADER_RE = re.compile(r"^Expected answer.*?:\s*$", re.IGNORECASE)
SECTION_DIVIDER_RE = re.compile(r"^[-=]{5,}\s*$")


def parse_evaluation_dataset(path: Path) -> List[Dict[str, str]]:
    """Parse the flat-text dataset into a list of {qid, question, expected_answer}.

    Format expected (matching evaluation_dataset.txt):

        ----------------
        Q1. <question, possibly multi-line>
        ----------------
        Expected answer (...):
            <indented expected answer, multi-line>

        Expected source mission tag: ...
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    entries: List[Dict[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].rstrip()
        m = QUESTION_HEADER_RE.match(line.strip())
        if not m:
            i += 1
            continue

        qid = f"Q{m.group(1)}"
        question_parts = [m.group(2).strip()]
        i += 1
        # Continuation lines for the question (until divider or "Expected answer")
        while i < n:
            nxt = lines[i].rstrip()
            if SECTION_DIVIDER_RE.match(nxt) or EXPECTED_HEADER_RE.match(nxt.strip()):
                break
            stripped = nxt.strip()
            if stripped:
                question_parts.append(stripped)
            i += 1
        # Skip divider line(s)
        while i < n and SECTION_DIVIDER_RE.match(lines[i].rstrip()):
            i += 1

        # Expected answer block (optional)
        expected: List[str] = []
        if i < n and EXPECTED_HEADER_RE.match(lines[i].strip()):
            i += 1
            while i < n:
                nxt = lines[i]
                stripped = nxt.strip()
                if not stripped:
                    # blank line ends the indented expected-answer block when the
                    # next non-blank line is unindented (e.g. "Expected source ...")
                    j = i + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n and not lines[j].startswith((" ", "\t")):
                        i = j
                        break
                    expected.append("")
                    i += 1
                    continue
                if not nxt.startswith((" ", "\t")):
                    break
                expected.append(stripped)
                i += 1

        entries.append({
            "qid": qid,
            "question": " ".join(question_parts).strip(),
            "expected_answer": " ".join(p for p in expected if p).strip(),
        })

    return entries


def run_pipeline_for_question(collection, openai_key: str, question: str,
                              k: int, model: str) -> Dict[str, object]:
    """Run retrieval + LLM for a single question and return contexts + answer."""
    docs_result = rag_client.retrieve_documents(collection, question, n_results=k)
    contexts: List[str] = []
    context_text = ""
    if docs_result and docs_result.get("documents"):
        contexts = list(docs_result["documents"][0])
        context_text = rag_client.format_context(
            docs_result["documents"][0], docs_result["metadatas"][0]
        )

    answer = llm_client.generate_response(
        openai_key=openai_key,
        user_message=question,
        context=context_text,
        conversation_history=[],
        model=model,
    )
    return {"answer": answer, "contexts": contexts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch RAGAS evaluation for the NASA RAG system.")
    parser.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "evaluation_dataset.txt"),
                        help="Path to evaluation dataset (default: evaluation_dataset.txt)")
    parser.add_argument("--chroma-dir", default=None,
                        help="ChromaDB directory (default: auto-discover)")
    parser.add_argument("--collection-name", default=None,
                        help="ChromaDB collection name (default: auto-discover)")
    parser.add_argument("--k", type=int, default=3, help="Documents to retrieve per question")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI chat model")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional cap on number of questions (0 = all)")
    parser.add_argument("--report-json", default="batch_report.json",
                        help="Path to write per-question + aggregate JSON report")
    parser.add_argument("--report-csv", default="batch_report.csv",
                        help="Path to write per-question CSV report")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY is not set (check .env).", file=sys.stderr)
        return 2

    base_url = os.environ.get("OPENAI_BASE_URL", "(default OpenAI)")
    print(f"Using OPENAI_BASE_URL = {base_url}")

    # 1. Load questions
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        return 2
    entries = parse_evaluation_dataset(dataset_path)
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]
    print(f"Loaded {len(entries)} questions from {dataset_path.name}")

    # 2. Resolve ChromaDB backend
    if args.chroma_dir and args.collection_name:
        chroma_dir, collection_name = args.chroma_dir, args.collection_name
    else:
        backends = rag_client.discover_chroma_backends()
        if not backends:
            print("ERROR: no ChromaDB backends discovered.", file=sys.stderr)
            return 2
        first = next(iter(backends.values()))
        chroma_dir = args.chroma_dir or first["directory"]
        collection_name = args.collection_name or first["collection_name"]
    print(f"Using collection: {chroma_dir} / {collection_name}")

    collection, ok, err = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not ok:
        print(f"ERROR: failed to initialise RAG: {err}", file=sys.stderr)
        return 2

    # 3. Run pipeline for every question, collect samples
    samples: List[Dict[str, object]] = []
    print("\n--- Running RAG pipeline per question ---")
    for entry in entries:
        print(f"  {entry['qid']}: {entry['question'][:80]}...")
        run = run_pipeline_for_question(
            collection, openai_key, entry["question"], args.k, args.model
        )
        samples.append({
            "qid": entry["qid"],
            "question": entry["question"],
            "expected_answer": entry["expected_answer"],
            "user_input": entry["question"],
            "response": run["answer"],
            "retrieved_contexts": run["contexts"],
        })

    # 4. Batch RAGAS evaluation in a single evaluate() call
    print("\n--- Running RAGAS batch evaluation ---")
    eval_input = [
        {"user_input": s["user_input"],
         "response": s["response"],
         "retrieved_contexts": s["retrieved_contexts"]}
        for s in samples
    ]
    result = ragas_evaluator.batch_evaluate_responses(eval_input)
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return 1

    per_sample = result["per_sample"]
    aggregate = result["aggregate"]

    # 5. Print per-question summary
    print("\n=== PER-QUESTION SCORES ===")
    metric_names = list(aggregate.keys())
    header = ["qid"] + metric_names
    print("  " + " | ".join(f"{h:>34}" if i else f"{h:<6}" for i, h in enumerate(header)))
    for sample, scores in zip(samples, per_sample):
        row = [sample["qid"]] + [
            f"{scores.get(m, float('nan')):.3f}" for m in metric_names
        ]
        print("  " + " | ".join(f"{c:>34}" if i else f"{c:<6}" for i, c in enumerate(row)))

    # 6. Aggregate summary
    print("\n=== AGGREGATE (mean across all questions) ===")
    for metric_name, mean_score in aggregate.items():
        print(f"  {metric_name:>40} : {mean_score:.4f}")

    # 7. Persist reports
    report_payload = {
        "config": {
            "dataset": str(dataset_path),
            "chroma_dir": chroma_dir,
            "collection_name": collection_name,
            "k": args.k,
            "model": args.model,
            "openai_base_url": base_url,
        },
        "aggregate": aggregate,
        "per_question": [
            {
                "qid": sample["qid"],
                "question": sample["question"],
                "answer": sample["response"],
                "n_contexts": len(sample["retrieved_contexts"]),
                "scores": scores,
            }
            for sample, scores in zip(samples, per_sample)
        ],
    }
    Path(args.report_json).write_text(json.dumps(report_payload, indent=2))
    print(f"\nWrote JSON report -> {args.report_json}")

    try:
        import csv
        with open(args.report_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["qid", "question"] + metric_names)
            for sample, scores in zip(samples, per_sample):
                writer.writerow(
                    [sample["qid"], sample["question"]]
                    + [scores.get(m, "") for m in metric_names]
                )
        print(f"Wrote CSV report  -> {args.report_csv}")
    except Exception as e:
        print(f"(CSV write skipped: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
