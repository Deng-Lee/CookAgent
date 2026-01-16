import unittest

from llm_utils import validate_extraction, validate_polish


class TestLLMUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence_set = {
            "parent_id": "recipes/test.md",
            "chunks": [
                {
                    "chunk_id": "c1",
                    "block_type": "operation",
                    "text": "小火慢炖 40 分钟，关火焖 5 分钟。",
                }
            ],
        }

    def test_validate_extraction_ok(self) -> None:
        extraction = {
            "intent": "ASK_TIME",
            "fields": {
                "time_info": [
                    {
                        "text": "慢炖 40 分钟",
                        "citations": [{"chunk_id": "c1", "quote": "慢炖 40 分钟"}],
                    }
                ]
            },
            "missing": [],
        }
        ok, errors = validate_extraction(extraction, self.evidence_set, expected_intent="ASK_TIME")
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_validate_extraction_bad_citation(self) -> None:
        extraction = {
            "intent": "ASK_TIME",
            "fields": {
                "time_info": [
                    {
                        "text": "慢炖 40 分钟",
                        "citations": [{"chunk_id": "c999", "quote": "慢炖 40 分钟"}],
                    }
                ]
            },
            "missing": [],
        }
        ok, errors = validate_extraction(extraction, self.evidence_set, expected_intent="ASK_TIME")
        self.assertFalse(ok)
        self.assertIn("citation_chunk_id_invalid", errors)

    def test_validate_extraction_number_out_of_bounds(self) -> None:
        extraction = {
            "intent": "ASK_TIME",
            "fields": {
                "time_info": [
                    {
                        "text": "慢炖 50 分钟",
                        "citations": [{"chunk_id": "c1", "quote": "慢炖 40 分钟"}],
                    }
                ]
            },
            "missing": [],
        }
        ok, errors = validate_extraction(extraction, self.evidence_set, expected_intent="ASK_TIME")
        self.assertFalse(ok)
        self.assertIn("number_out_of_bounds", errors)

    def test_validate_polish_ok(self) -> None:
        ok, errors = validate_polish("慢炖 40 分钟即可。", "大约 40 分钟即可。")
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_validate_polish_new_number(self) -> None:
        ok, errors = validate_polish("慢炖 40 分钟即可。", "慢炖 60 分钟即可。")
        self.assertFalse(ok)
        self.assertIn("number_out_of_bounds", errors)


if __name__ == "__main__":
    unittest.main()
