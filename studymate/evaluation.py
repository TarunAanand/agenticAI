import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

from studymate.groq_client import ChatMessage, GroqAPIError
from studymate.materials import TextChunk
from studymate.workflows import run_advisor_workflow


EVALUATION_CASES_PATH = (
    Path(__file__).resolve().parent.parent / "evaluations" / "cases.json"
)
EVALUATION_MODES = ("no_rag", "basic_rag", "agentic_rag")
WORD_PATTERN = re.compile(r"[a-z0-9]+")


class EvaluationCase(TypedDict):
    id: str
    question: str
    source: str
    content: str
    expected_terms: list[str]
    expected_source: str


def load_evaluation_cases(path: Path = EVALUATION_CASES_PATH) -> list[EvaluationCase]:
    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load evaluation dataset at {path}.") from error
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Evaluation dataset must contain at least one case.")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for case in raw_cases:
        if not isinstance(case, dict):
            raise ValueError("Evaluation dataset contains an invalid case.")
        case_id = case.get("id")
        question = case.get("question")
        source = case.get("source")
        content = case.get("content")
        expected_terms = case.get("expected_terms")
        expected_source = case.get("expected_source")
        if (
            not all(isinstance(value, str) and value.strip() for value in (
                case_id, question, source, content, expected_source
            ))
            or not isinstance(expected_terms, list)
            or not expected_terms
            or any(not isinstance(term, str) or not term.strip() for term in expected_terms)
        ):
            raise ValueError("Evaluation case is missing a question, source, text, or expected terms.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        seen_ids.add(case_id)
        cases.append(
            {
                "id": case_id,
                "question": question,
                "source": source,
                "content": content,
                "expected_terms": expected_terms,
                "expected_source": expected_source,
            }
        )
    return cases


def _contains_expected_term(answer: str, expected_term: str) -> bool:
    answer_tokens = WORD_PATTERN.findall(answer.casefold())
    term_tokens = WORD_PATTERN.findall(expected_term.casefold())
    if not term_tokens:
        return False
    answer_text = f" {' '.join(answer_tokens)} "
    term_text = f" {' '.join(term_tokens)} "
    return term_text in answer_text


def evaluate_model_modes(
    cases: list[EvaluationCase],
    models: list[str],
) -> dict:
    """Run every evaluation case against no/basic/agentic retrieval for each model."""
    clean_models = list(dict.fromkeys(model.strip() for model in models if model.strip()))
    if not clean_models:
        raise ValueError("Provide at least one model name.")

    results: list[dict] = []
    for model in clean_models:
        for mode in EVALUATION_MODES:
            for case in cases:
                chunks: list[TextChunk] = (
                    [{"source": case["source"], "content": case["content"]}]
                    if mode != "no_rag"
                    else []
                )
                history: list[ChatMessage] = [
                    {"role": "user", "content": case["question"]}
                ]
                started_at = time.perf_counter()
                try:
                    state = run_advisor_workflow(
                        question=case["question"],
                        history=history,
                        chunks=chunks,
                        student_context="",
                        agentic=mode == "agentic_rag",
                        model=model,
                    )
                    answer = state["reply"]
                    retrieved_sources = sorted(
                        {chunk["source"] for chunk in state.get("retrieved", [])}
                    )
                    error = None
                except GroqAPIError as exception:
                    answer = ""
                    retrieved_sources = []
                    error = str(exception)
                elapsed_ms = round((time.perf_counter() - started_at) * 1_000, 1)
                matched_terms = [
                    term
                    for term in case["expected_terms"]
                    if _contains_expected_term(answer, term)
                ]
                results.append(
                    {
                        "model": model,
                        "mode": mode,
                        "case_id": case["id"],
                        "latency_ms": elapsed_ms,
                        "expected_term_recall": len(matched_terms) / len(case["expected_terms"]),
                        "matched_terms": matched_terms,
                        "retrieval_hit": (
                            case["expected_source"] in retrieved_sources
                            if mode != "no_rag"
                            else None
                        ),
                        "citation_present": "[1]" in answer if mode != "no_rag" else None,
                        "answer": answer,
                        "error": error,
                    }
                )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in results:
        grouped[(result["model"], result["mode"])].append(result)
    summaries = []
    for (model, mode), group in grouped.items():
        summary = {
            "model": model,
            "mode": mode,
            "cases": len(group),
            "mean_expected_term_recall": round(
                sum(result["expected_term_recall"] for result in group) / len(group), 3
            ),
            "mean_latency_ms": round(
                sum(result["latency_ms"] for result in group) / len(group), 1
            ),
            "errors": sum(result["error"] is not None for result in group),
        }
        if mode != "no_rag":
            summary["retrieval_hit_rate"] = round(
                sum(result["retrieval_hit"] for result in group) / len(group), 3
            )
            summary["citation_rate"] = round(
                sum(result["citation_present"] for result in group) / len(group), 3
            )
        summaries.append(summary)
    return {
        "dataset_case_count": len(cases),
        "models": clean_models,
        "modes": list(EVALUATION_MODES),
        "provider_calls_expected": sum(
            1 if result["mode"] in {"no_rag", "basic_rag"} else 3
            for result in results
        ),
        "summaries": summaries,
        "results": results,
        "metric_notes": {
            "expected_term_recall": (
                "Fraction of hand-authored expected phrases appearing in the answer; "
                "a quick lexical proxy, not semantic grading."
            ),
            "retrieval_hit_rate": (
                "Fraction of RAG cases where the expected source file appears in retrieved chunks."
            ),
            "citation_rate": "Fraction of RAG answers containing a [1] source citation.",
            "agentic_provider_calls": (
                "Up to three per case when source chunks are present: query planner, answer, critic."
            ),
        },
    }
