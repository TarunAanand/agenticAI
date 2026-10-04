import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studymate.config import get_groq_model
from studymate.evaluation import EVALUATION_MODES, evaluate_model_modes, load_evaluation_cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare StudyMate's no-RAG, basic-RAG, and agentic-RAG modes."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make real Groq API calls; omitted by default to show only the run plan.",
    )
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated Groq model names. Defaults to GROQ_EVAL_MODELS, "
            "then GROQ_MODEL/the app's default."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a JSON report. Without this, no report file is written.",
    )
    args = parser.parse_args()

    configured_models = args.models or os.getenv("GROQ_EVAL_MODELS", "")
    models = list(
        dict.fromkeys(
            model.strip()
            for model in (
                configured_models.split(",")
                if configured_models.strip()
                else [get_groq_model()]
            )
            if model.strip()
        )
    )
    if len(models) > 3:
        parser.error("Select at most 3 models.")

    cases = load_evaluation_cases()
    expected_calls = len(cases) * len(models) * 5
    if not args.live:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "dataset_cases": len(cases),
                    "models": models,
                    "modes": list(EVALUATION_MODES),
                    "estimated_provider_calls": expected_calls,
                    "next_step": "Re-run with --live to call Groq and generate metrics.",
                },
                indent=2,
            )
        )
        return 0

    if len(models) < 2:
        parser.error(
            "Live comparison requires 2 or 3 models. Pass --models model-a,model-b "
            "or configure GROQ_EVAL_MODELS."
        )

    report = evaluate_model_modes(cases, models)
    rendered_report = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered_report + "\n", encoding="utf-8")
    print(rendered_report)
    return 1 if any(item["error"] for item in report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
