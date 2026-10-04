import unittest

from studymate.quiz import QuizError, QuizQuestion, score_quiz, validate_quiz_data


class QuizTests(unittest.TestCase):
    def setUp(self):
        self.raw = [
            {
                "question": "What is a stack?",
                "options": [
                    "A last-in-first-out structure",
                    "A first-in-first-out structure",
                    "A sorted tree",
                    "A hash function",
                ],
                "answer_index": 0,
                "explanation": "A stack removes the most recently added item first.",
                "source_ids": [],
            },
            {
                "question": "Which structure is first-in-first-out?",
                "options": ["Stack", "Queue", "Heap", "Graph"],
                "answer_index": 1,
                "explanation": "A queue removes items in insertion order.",
                "source_ids": [],
            },
        ]

    def test_validation_accepts_quiz_questions(self):
        questions = validate_quiz_data(self.raw, expected_count=2, source_count=0)
        self.assertEqual(questions[0]["question"], "What is a stack?")

    def test_validation_rejects_invalid_answer_and_source_ids(self):
        invalid_answer = [{**self.raw[0], "answer_index": True}]
        with self.assertRaisesRegex(QuizError, "answer key"):
            validate_quiz_data(invalid_answer, expected_count=1, source_count=0)

        invalid_source = [{**self.raw[0], "source_ids": [2]}]
        with self.assertRaisesRegex(QuizError, "invalid source"):
            validate_quiz_data(invalid_source, expected_count=1, source_count=1)

    def test_validation_requires_distinct_options_and_exact_count(self):
        duplicate_options = [
            {**self.raw[0], "options": ["Same", "Same", "Third", "Fourth"]}
        ]
        with self.assertRaisesRegex(QuizError, "four distinct"):
            validate_quiz_data(duplicate_options, expected_count=1, source_count=0)
        with self.assertRaisesRegex(QuizError, "wrong number"):
            validate_quiz_data(self.raw, expected_count=3, source_count=0)

    def test_scoring_handles_correct_incorrect_and_skipped_answers(self):
        questions: list[QuizQuestion] = validate_quiz_data(
            self.raw, expected_count=2, source_count=0
        )
        result = score_quiz(questions, [0, None])

        self.assertEqual(
            result,
            [
                {"is_correct": True, "selected_index": 0, "correct_index": 0},
                {"is_correct": False, "selected_index": None, "correct_index": 1},
            ],
        )


if __name__ == "__main__":
    unittest.main()
