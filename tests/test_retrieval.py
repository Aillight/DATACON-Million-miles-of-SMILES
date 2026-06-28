import unittest

from backend.parsing.chunking import TextChunk
from backend.retrieval import make_huggingface_embedding_function, rank_chunks_for_extraction


class RetrievalTests(unittest.TestCase):
    def test_rank_chunks_for_nanozymes_prioritizes_particle_and_kinetic_terms(self) -> None:
        chunks = [
            TextChunk(
                index=0,
                section_title="Results",
                text="The catalyst was reused in five cycles and the discussion focuses on yield.",
                start_char=0,
                end_char=80,
            ),
            TextChunk(
                index=1,
                section_title="Experimental",
                text="Mn3O4 nanoparticles had a particle diameter of 10 nm with Km 0.027 mM and Vmax 126.7 nM s-1.",
                start_char=81,
                end_char=180,
            ),
            TextChunk(
                index=2,
                section_title="Conclusion",
                text="The material is promising for catalysis.",
                start_char=181,
                end_char=230,
            ),
        ]

        ranked = rank_chunks_for_extraction(chunks, domain="Nanozymes", top_k=1)

        self.assertEqual(1, ranked[0].chunk.index)
        self.assertGreater(ranked[0].score, 0)
        self.assertIn("nanoparticles", ranked[0].matched_terms)
        self.assertIn("vmax", ranked[0].matched_terms)

    def test_rank_chunks_returns_empty_for_zero_top_k(self) -> None:
        chunks = [TextChunk(index=0, section_title="Results", text="MIC 2 ug/ml", start_char=0, end_char=10)]

        self.assertEqual([], rank_chunks_for_extraction(chunks, domain="Oxazolidinones", top_k=0))

    def test_hybrid_retrieval_uses_embedding_scores_when_available(self) -> None:
        chunks = [
            TextChunk(index=0, section_title="Results", text="generic catalyst reuse", start_char=0, end_char=10),
            TextChunk(index=1, section_title="Results", text="semantic target without query words", start_char=11, end_char=50),
        ]

        def fake_embeddings(_texts: list[str]) -> list[list[float]]:
            return [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]

        ranked = rank_chunks_for_extraction(
            chunks,
            domain="Nanozymes",
            top_k=1,
            retrieval_mode="hybrid",
            embedding_fn=fake_embeddings,
        )

        self.assertEqual(1, ranked[0].chunk.index)
        self.assertEqual("hybrid", ranked[0].retrieval_method)
        self.assertEqual(1.0, ranked[0].embedding_score)

    def test_hybrid_retrieval_falls_back_to_tfidf_when_embeddings_fail(self) -> None:
        chunks = [
            TextChunk(index=0, section_title="Results", text="nanoparticles Vmax Km", start_char=0, end_char=20),
        ]

        def failing_embeddings(_texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding backend down")

        ranked = rank_chunks_for_extraction(
            chunks,
            domain="Nanozymes",
            top_k=1,
            retrieval_mode="hybrid",
            embedding_fn=failing_embeddings,
        )

        self.assertEqual("tfidf_fallback", ranked[0].retrieval_method)
        self.assertIsNone(ranked[0].embedding_score)

    def test_make_huggingface_embedding_function_normalizes_client_response(self) -> None:
        class FakeClient:
            def feature_extraction(self, texts, normalize, truncate, model):
                self.request = {
                    "texts": texts,
                    "normalize": normalize,
                    "truncate": truncate,
                    "model": model,
                }
                return [[1, 0], [0, 1]]

        client = FakeClient()
        embed = make_huggingface_embedding_function(model="embed-test", token="token", client=client)

        self.assertEqual([[1.0, 0.0], [0.0, 1.0]], embed(["a", "b"]))
        self.assertEqual("embed-test", client.request["model"])
        self.assertTrue(client.request["normalize"])


if __name__ == "__main__":
    unittest.main()
