import tempfile
import unittest
from pathlib import Path

from studymate.storage import (
    add_focus_topic,
    delete_focus_topic,
    get_focus_topics,
    get_active_study_plan,
    get_student_profile,
    get_subject_progress,
    initialize_database,
    save_student_profile,
    save_subject_progress,
    approve_study_plan,
    set_study_session_status,
)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "nested" / "test.sqlite3"
        initialize_database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_seeded_profile_progress_and_topics_persist(self):
        profile = get_student_profile(self.db_path)
        self.assertEqual(profile["first_name"], "Alex")
        self.assertEqual(len(get_subject_progress(self.db_path)), 3)
        self.assertEqual(len(get_focus_topics(self.db_path)), 3)

        save_student_profile(
            "Sam",
            "Physics",
            "Spring 2027",
            "Worked examples",
            "Preparing for graduate study",
            self.db_path,
        )
        profile = get_student_profile(self.db_path)
        self.assertEqual(profile["first_name"], "Sam")
        self.assertEqual(profile["advisor_memory"], "Preparing for graduate study")

    def test_subject_progress_is_validated_and_persisted(self):
        save_subject_progress(
            "Data Structures",
            91,
            "Review trees",
            7,
            self.db_path,
        )
        subject = next(
            item for item in get_subject_progress(self.db_path)
            if item["name"] == "Data Structures"
        )
        self.assertEqual(subject["progress"], 91)
        self.assertEqual(subject["next_step"], "Review trees")
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            save_subject_progress("Data Structures", 101, "", 2, self.db_path)

    def test_focus_topics_can_be_added_deduplicated_and_removed(self):
        added = add_focus_topic("Optics", "Physics", self.db_path)
        duplicate = add_focus_topic("Optics", "Physics", self.db_path)
        topics = get_focus_topics(self.db_path)
        topic = next(item for item in topics if item["name"] == "Optics")

        self.assertTrue(added)
        self.assertFalse(duplicate)
        delete_focus_topic(topic["id"], self.db_path)
        self.assertFalse(any(item["name"] == "Optics" for item in get_focus_topics(self.db_path)))

    def test_profile_required_fields_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "required"):
            save_student_profile("", "Physics", "Spring 2027", "", "", self.db_path)

    def test_approved_plan_persists_and_sessions_can_be_completed(self):
        draft = [
            {
                "study_date": "2026-10-05",
                "subject": "Data Structures",
                "task": "Review graph traversal",
                "duration_minutes": 45,
            },
            {
                "study_date": "2026-10-06",
                "subject": "Linear Algebra",
                "task": "Practice eigenvectors",
                "duration_minutes": 30,
            },
        ]
        approve_study_plan(draft, self.db_path)
        plan = get_active_study_plan(self.db_path)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["status"], "planned")

        set_study_session_status(plan[0]["id"], "completed", self.db_path)
        updated_plan = get_active_study_plan(self.db_path)
        self.assertEqual(updated_plan[0]["status"], "completed")

    def test_invalid_plan_is_not_approved(self):
        with self.assertRaisesRegex(ValueError, "valid ISO date"):
            approve_study_plan(
                [
                    {
                        "study_date": "tomorrow",
                        "subject": "Data Structures",
                        "task": "Study",
                        "duration_minutes": 45,
                    }
                ],
                self.db_path,
            )
        self.assertEqual(get_active_study_plan(self.db_path), [])


if __name__ == "__main__":
    unittest.main()
