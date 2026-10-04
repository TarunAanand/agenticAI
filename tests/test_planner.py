import unittest
from datetime import date

from studymate.planner import build_weekly_draft
from studymate.storage import FocusTopic, SubjectProgress


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.subjects: list[SubjectProgress] = [
            {
                "name": "Data Structures",
                "progress": 78,
                "next_step": "Review graph traversal",
                "weekly_hours": 6,
            },
            {
                "name": "Linear Algebra",
                "progress": 56,
                "next_step": "Practice eigenvalue problems",
                "weekly_hours": 4,
            },
        ]
        self.topics: list[FocusTopic] = [
            {"id": 1, "name": "Eigenvectors", "subject": "Linear Algebra"},
        ]

    def test_draft_uses_only_selected_days_and_obeys_daily_budget(self):
        draft = build_weekly_draft(
            self.subjects,
            self.topics,
            study_days=[0, 2],
            minutes_per_day=90,
            start_date=date(2026, 10, 4),
        )

        self.assertEqual({item["study_date"] for item in draft}, {"2026-10-05", "2026-10-07"})
        for study_date in {item["study_date"] for item in draft}:
            daily_minutes = sum(
                item["duration_minutes"] for item in draft if item["study_date"] == study_date
            )
            self.assertLessEqual(daily_minutes, 90)
        self.assertTrue(all(item["duration_minutes"] >= 25 for item in draft))

    def test_soonest_deadline_is_prioritized(self):
        draft = build_weekly_draft(
            self.subjects,
            self.topics,
            study_days=[0],
            minutes_per_day=45,
            start_date=date(2026, 10, 4),
        )
        self.assertEqual(draft[0]["subject"], "Data Structures")

    def test_empty_days_and_invalid_time_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_weekly_draft(self.subjects, self.topics, [], 60, date(2026, 10, 4))
        with self.assertRaisesRegex(ValueError, "between 25 and 240"):
            build_weekly_draft(self.subjects, self.topics, [0], 20, date(2026, 10, 4))

    def test_overdue_sample_deadline_does_not_keep_priority(self):
        self.subjects = [
            {
                "name": "Data Structures",
                "progress": 80,
                "next_step": "Review trees",
                "weekly_hours": 3,
            },
            {
                "name": "Linear Algebra",
                "progress": 50,
                "next_step": "Practice vectors",
                "weekly_hours": 4,
            },
        ]
        draft = build_weekly_draft(
            self.subjects,
            [],
            study_days=[0],
            minutes_per_day=45,
            start_date=date(2026, 10, 20),
        )
        self.assertEqual(draft[0]["subject"], "Linear Algebra")


if __name__ == "__main__":
    unittest.main()
