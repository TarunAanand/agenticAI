import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studymate.evaluation import (
    EVALUATION_MODES,
    evaluate_model_modes,
    load_evaluation_cases,
)


class EvaluationTests(unittest.TestCase):
    def test_dataset_contains_unique_well_formed_cases(self):
        cases = load_evaluation_cases()

        self.assertGreaterEqual(len(cases), 3)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(all(case["expected_terms"] for case in cases))

    def test_evaluation_compares_modes_and_models_and_aggregates_metrics(self):
        cases = load_evaluation_cases()[:2]

        def fake_workflow(**kwargs):
            uses_rag = bool(kwargs["chunks"])
            return {
                "reply": (
                    "last in first out lifo most recently pushed light energy "
                    "chemical energy glucose direction is unchanged eigenvalue scaled "
                    "duplicated data anomalies insertion"
                ),
                "retrieved": (
                    [
                        {
                            "source": kwargs["chunks"][0]["source"],
                            "content": kwargs["chunks"][0]["content"],
                            "score": 1.0,
                        }
                    ]
                    if uses_rag
                    else []
                ),
            }

        with patch(
            "studymate.evaluation.run_advisor_workflow",
            side_effect=fake_workflow,
        ) as run_workflow:
            report = evaluate_model_modes(cases, ["model-a", "model-b"])

        self.assertEqual(report["models"], ["model-a", "model-b"])
        self.assertEqual(report["modes"], list(EVALUATION_MODES))
        self.assertEqual(report["provider_calls_expected"], 2 * 2 * 5)
        self.assertEqual(len(report["results"]), 2 * 2 * 3)
        self.assertEqual(len(report["summaries"]), 2 * 3)
        self.assertEqual(run_workflow.call_count, len(report["results"]))
        self.assertEqual(
            {call.kwargs["model"] for call in run_workflow.call_args_list},
            {"model-a", "model-b"},
        )
        self.assertTrue(all(summary["errors"] == 0 for summary in report["summaries"]))

    def test_invalid_dataset_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = Path(temporary_directory) / "empty.json"
            fixture.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one"):
                load_evaluation_cases(fixture)


if __name__ == "__main__":
    unittest.main()
