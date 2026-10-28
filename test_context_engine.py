import unittest
from context_engine import SourcePassage, build_context_prompt, assess_answer, collect_references, tafsir_comparison

class ContextEngineTests(unittest.TestCase):
    def test_context_prompt_numbers_sources(self):
        p = SourcePassage("x","translation","Translation","Text",surah=2,ayah=255)
        prompt = build_context_prompt("Explain this", [p])
        self.assertIn("[SOURCE 1]", prompt)
        self.assertIn("Quran 2:255", prompt)

    def test_missing_citation_warns(self):
        p = SourcePassage("x","tafsir","Tafsir","Commentary")
        self.assertTrue(assess_answer("An uncited answer", [p]))

    def test_cross_reference_parser(self):
        self.assertEqual(collect_references("See 2:255 and 3:18; then 2:255"), [(2,255),(3,18)])

    def test_tafsir_comparison_preserves_author(self):
        bundle = {"surah":2,"ayah":255,"tafsir":[{"book":{"id":4,"name":"Tafsir Example","author":{"full_name":"A Scholar"}},"content":[{"text":"Commentary text"}]}]}
        result = tafsir_comparison(bundle)
        self.assertEqual(result[0]["author"], "A Scholar")
        self.assertIn("Commentary text", result[0]["text"])

if __name__ == "__main__":
    unittest.main()
