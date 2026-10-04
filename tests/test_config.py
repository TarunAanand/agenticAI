import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from studymate.config import get_groq_api_key


class ConfigTests(unittest.TestCase):
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_reading_api_key_does_not_print_it(self):
        output = io.StringIO()

        with redirect_stdout(output):
            key = get_groq_api_key()

        self.assertEqual(key, "test-key")
        self.assertEqual(output.getvalue(), "")
