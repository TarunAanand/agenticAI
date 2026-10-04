import unittest
from unittest.mock import patch

from studymate.workflows import run_advisor_workflow, run_quiz_workflow


class WorkflowTests(unittest.TestCase):
    def test_basic_advisor_flow_retrieves_then_answers_without_agent_nodes(self):
        chunk = {"source": "notes.txt", "content": "Stacks use last in first out."}
        retrieved = {**chunk, "score": 0.7}
        history = [{"role": "user", "content": "How does a stack work?"}]
        with (
            patch(
                "studymate.workflows.retrieve_chunks",
                return_value=[retrieved],
            ) as retrieve,
            patch(
                "studymate.workflows.get_chat_reply",
                return_value="A stack uses LIFO [1].",
            ) as answer,
            patch("studymate.workflows.plan_retrieval_queries") as plan,
            patch("studymate.workflows.critique_grounded_answer") as critic,
        ):
            result = run_advisor_workflow(
                question=history[0]["content"],
                history=history,
                chunks=[chunk],
                student_context="Program: CS",
                agentic=False,
            )

        self.assertEqual(result["reply"], "A stack uses LIFO [1].")
        self.assertEqual(result["retrieved"], [retrieved])
        self.assertEqual(result["search_queries"], [history[0]["content"]])
        retrieve.assert_called_once()
        answer.assert_called_once()
        plan.assert_not_called()
        critic.assert_not_called()

    def test_agentic_advisor_flow_plans_search_and_critiques_answer(self):
        retrieved = [
            {
                "source": "notes.txt",
                "content": "Stacks use last in first out.",
                "score": 0.8,
            }
        ]
        with (
            patch(
                "studymate.workflows.plan_retrieval_queries",
                return_value=["stack LIFO behavior"],
            ) as plan,
            patch(
                "studymate.workflows.retrieve_chunks",
                return_value=retrieved,
            ) as retrieve,
            patch(
                "studymate.workflows.get_chat_reply",
                return_value="A stack uses LIFO.",
            ) as answer,
            patch(
                "studymate.workflows.critique_grounded_answer",
                return_value={"verdict": "supported", "answer": "A stack uses LIFO [1]."},
            ) as critic,
        ):
            result = run_advisor_workflow(
                question="How does a stack work?",
                history=[{"role": "user", "content": "How does a stack work?"}],
                chunks=[{"source": "notes.txt", "content": "Stacks use last in first out."}],
                student_context="",
                agentic=True,
            )

        self.assertEqual(result["search_queries"], ["How does a stack work?", "stack LIFO behavior"])
        self.assertEqual(result["critique_verdict"], "supported")
        self.assertEqual(result["reply"], "A stack uses LIFO [1].")
        plan.assert_called_once()
        self.assertEqual(retrieve.call_count, 2)
        answer.assert_called_once()
        critic.assert_called_once()

    def test_quiz_flow_retrieves_context_before_generating(self):
        retrieved = [
            {
                "source": "notes.txt",
                "content": "Stacks use last in first out.",
                "score": 0.8,
            }
        ]
        quiz = [
            {
                "question": "How does a stack work?",
                "options": ["LIFO", "FIFO", "Sorted", "Random"],
                "answer_index": 0,
                "explanation": "A stack is LIFO.",
                "source_ids": [1],
            }
        ]
        with (
            patch("studymate.workflows.retrieve_chunks", return_value=retrieved) as retrieve,
            patch("studymate.workflows.generate_quiz", return_value=quiz) as generate,
        ):
            result = run_quiz_workflow(
                topic="Stacks",
                question_count=3,
                chunks=[{"source": "notes.txt", "content": "Stacks use last in first out."}],
            )

        self.assertEqual(result["questions"], quiz)
        self.assertEqual(result["retrieved"], retrieved)
        retrieve.assert_called_once()
        generate.assert_called_once_with("Stacks", 3, retrieved, model=None)


if __name__ == "__main__":
    unittest.main()
