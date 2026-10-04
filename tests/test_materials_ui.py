import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class MaterialsUiTests(unittest.TestCase):
    def test_uploaded_material_can_be_removed_without_being_readded(self):
        quiet = io.StringIO()
        with contextlib.redirect_stderr(quiet):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.radio[0].set_value("Materials").run()
            app.file_uploader[0].upload(
                "notes.txt",
                b"Stacks use last in first out.",
                "text/plain",
            ).run()

            self.assertFalse(app.exception, app.exception)
            self.assertEqual(len(app.session_state["materials_documents"]), 1)
            self.assertEqual(len(app.session_state["materials_chunks"]), 1)
            uploader_generation_after_upload = app.session_state[
                "materials_uploader_generation"
            ]

            next(button for button in app.button if button.label == "Remove").click().run()

            self.assertFalse(app.exception, app.exception)
            self.assertEqual(app.session_state["materials_documents"], [])
            self.assertEqual(app.session_state["materials_chunks"], [])
            self.assertGreater(
                app.session_state["materials_uploader_generation"],
                uploader_generation_after_upload,
            )

    def test_low_relevance_query_does_not_send_material_context(self):
        with (
            patch("studymate.config.get_groq_api_key", return_value="test-key"),
            patch(
                "studymate.workflows.get_chat_reply",
                return_value="General answer.",
            ) as answer,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.radio[0].set_value("AI Advisor").run()
            app.session_state["materials_documents"] = [
                {"id": "doc", "name": "notes.txt", "size": 40, "chunk_count": 1}
            ]
            app.session_state["materials_chunks"] = [
                {
                    "source": "notes.txt",
                    "content": "Only a weak lexical match.",
                    "document_id": "doc",
                }
            ]
            app.chat_input[0].set_value("Question unrelated to the material").run()

            self.assertFalse(app.exception, app.exception)
            answer.assert_called_once()
            self.assertEqual(answer.call_args.args[1], [])
            self.assertTrue(
                any(
                    "No close match was found" in element.value
                    for element in app.info
                )
            )


if __name__ == "__main__":
    unittest.main()
