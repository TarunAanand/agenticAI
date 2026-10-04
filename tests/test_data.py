import unittest
from datetime import date

from studymate.data import get_upcoming_deadlines


class DeadlineTests(unittest.TestCase):
    def test_deadlines_are_filtered_and_sorted_in_requested_horizon(self):
        deadlines = get_upcoming_deadlines(as_of=date(2026, 10, 10), days=6)

        self.assertEqual(
            [(item["title"], item["due"]) for item in deadlines],
            [
                ("Eigenvalues Quiz", "Oct 11"),
                ("Database Design Project", "Oct 16"),
            ],
        )

    def test_overdue_deadlines_are_not_presented_as_upcoming(self):
        self.assertEqual(get_upcoming_deadlines(as_of=date(2026, 10, 20)), [])

    def test_negative_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            get_upcoming_deadlines(as_of=date(2026, 10, 4), days=-1)


if __name__ == "__main__":
    unittest.main()
