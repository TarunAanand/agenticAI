from typing import TypedDict


class QuizQuestion(TypedDict):
    question: str
    options: list[str]
    answer_index: int
    explanation: str
    source_ids: list[int]


class QuizResult(TypedDict):
    is_correct: bool
    selected_index: int | None
    correct_index: int


class QuizError(ValueError):
    """Invalid or incomplete quiz data returned by the model."""


def validate_quiz_data(
    raw: object,
    expected_count: int,
    source_count: int,
) -> list[QuizQuestion]:
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise QuizError("The quiz service returned the wrong number of questions.")

    questions: list[QuizQuestion] = []
    for item in raw:
        if not isinstance(item, dict):
            raise QuizError("The quiz service returned an invalid question.")
        question = item.get("question")
        options = item.get("options")
        answer_index = item.get("answer_index")
        explanation = item.get("explanation")
        source_ids = item.get("source_ids", [])
        if not isinstance(question, str) or not question.strip() or len(question) > 800:
            raise QuizError("A quiz question is missing or too long.")
        if (
            not isinstance(options, list)
            or len(options) != 4
            or any(not isinstance(option, str) or not option.strip() or len(option) > 300 for option in options)
            or len({option.strip().casefold() for option in options}) != 4
        ):
            raise QuizError("Each quiz question must have four distinct, non-empty options.")
        if (
            not isinstance(answer_index, int)
            or isinstance(answer_index, bool)
            or not 0 <= answer_index < len(options)
        ):
            raise QuizError("A quiz question has an invalid answer key.")
        if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 1_500:
            raise QuizError("A quiz question is missing its explanation.")
        if (
            not isinstance(source_ids, list)
            or any(
                not isinstance(source_id, int)
                or isinstance(source_id, bool)
                or source_id < 1
                or source_id > source_count
                for source_id in source_ids
            )
        ):
            raise QuizError("A quiz question references an invalid source.")
        questions.append(
            {
                "question": question.strip(),
                "options": [option.strip() for option in options],
                "answer_index": answer_index,
                "explanation": explanation.strip(),
                "source_ids": list(dict.fromkeys(source_ids)),
            }
        )
    return questions


def score_quiz(
    questions: list[QuizQuestion],
    selected_answers: list[int | None],
) -> list[QuizResult]:
    if len(questions) != len(selected_answers):
        raise ValueError("Provide one selected answer for each quiz question.")
    return [
        {
            "is_correct": selected == question["answer_index"],
            "selected_index": selected,
            "correct_index": question["answer_index"],
        }
        for question, selected in zip(questions, selected_answers)
    ]
