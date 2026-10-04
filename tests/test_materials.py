from io import BytesIO
import unittest

from pypdf import PdfWriter

from studymate.materials import (
    CHUNK_SIZE,
    MAX_FILE_BYTES,
    MaterialError,
    extract_and_chunk,
    format_context,
    retrieve_chunks,
)


class MaterialsTests(unittest.TestCase):
    def test_text_upload_is_chunked_with_source_metadata(self):
        chunks = extract_and_chunk("lecture.txt", b"Linear algebra studies vectors and matrices.")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source"], "lecture.txt")
        self.assertIn("vectors", chunks[0]["content"])

    def test_long_text_is_split_with_overlapping_context(self):
        text = ("eigenvalues explain matrix transformations " * 100).encode()
        chunks = extract_and_chunk("notes.md", text)

        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(len(chunk["content"]) for chunk in chunks), CHUNK_SIZE)
        self.assertTrue(set(chunks[0]["content"].split()[-10:]) & set(chunks[1]["content"].split()[:10]))

    def test_retrieval_ranks_the_matching_material_first(self):
        chunks = [
            {"source": "biology.txt", "content": "Photosynthesis converts sunlight into chemical energy."},
            {"source": "history.txt", "content": "The industrial revolution changed manufacturing."},
            {"source": "biology.txt", "content": "Chlorophyll absorbs light during photosynthesis."},
        ]

        result = retrieve_chunks(
            "How does chlorophyll relate to photosynthesis?", chunks, top_k=2
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "biology.txt")
        self.assertIn("Chlorophyll", result[0]["content"])
        self.assertGreater(result[0]["score"], 0)
        self.assertIn("[1] Source: biology.txt", format_context(result))

    def test_retrieval_drops_weak_single_term_matches(self):
        content = " ".join(f"unrelated{i}" for i in range(100)) + " chlorophyll"
        chunks = [{"source": "long.txt", "content": content}]

        self.assertEqual(retrieve_chunks("chlorophyll?", chunks), [])

    def test_retrieval_threshold_is_validated(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            retrieve_chunks(
                "chlorophyll",
                [{"source": "biology.txt", "content": "chlorophyll absorbs light"}],
                minimum_score=1.1,
            )

    def test_unsupported_empty_and_oversized_uploads_fail_clearly(self):
        with self.assertRaisesRegex(MaterialError, "Unsupported"):
            extract_and_chunk("slides.pptx", b"data")
        with self.assertRaisesRegex(MaterialError, "empty"):
            extract_and_chunk("notes.txt", b"")
        with self.assertRaisesRegex(MaterialError, "8 MB"):
            extract_and_chunk("notes.txt", b"x" * (MAX_FILE_BYTES + 1))

    def test_pdf_with_no_selectable_text_is_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        pdf_bytes = BytesIO()
        writer.write(pdf_bytes)
        with self.assertRaisesRegex(MaterialError, "No selectable text"):
            extract_and_chunk("blank.pdf", pdf_bytes.getvalue())


if __name__ == "__main__":
    unittest.main()
