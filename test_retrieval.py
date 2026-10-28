import unittest
from retrieval import Evidence, HybridRetriever, normalize_search_results, build_evidence_prompt, validate_citations

class RetrievalTests(unittest.TestCase):
    def test_normalizes_search_results(self):
        rows = normalize_search_results({"results":[{"id":1,"title":"Tafsir A","content":"Patience and trust","surah":2,"ayah":153}]}, "patience")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].title, "Tafsir A")
        self.assertIn("Patience", rows[0].text)

    def test_lexical_fallback_ranks_match(self):
        r = HybridRetriever()
        r.add([
            Evidence("1","tafsir","Charity","Give to people in need"),
            Evidence("2","tafsir","Patience","Be patient during hardship"),
        ])
        results = r.search("patience hardship", top_k=2)
        self.assertEqual(results[0].source_id, "2")

    def test_prompt_includes_sources(self):
        evidence = [Evidence("1","translation","Translation","Do good",surah=2,ayah=177)]
        prompt = build_evidence_prompt("What is good?", evidence)
        self.assertIn("[SOURCE 1]", prompt)
        self.assertIn("Quran 2:177", prompt)

    def test_citation_warning(self):
        evidence = [Evidence("1","translation","Translation","Text")]
        self.assertTrue(validate_citations("This is an answer without a source.", evidence))

if __name__ == "__main__":
    unittest.main()
