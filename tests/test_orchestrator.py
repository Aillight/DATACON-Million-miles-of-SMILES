import unittest

from agent.core import run_agent_task


class OrchestratorTests(unittest.TestCase):
    def test_default_orchestrator_returns_trace_and_context(self) -> None:
        result = run_agent_task(
            "Find synthesis evidence.",
            {
                "text": "Experimental Section\n\nSynthesis was performed in ethanol.",
                "target_sections": ["experimental section"],
                "max_chars": 500,
                "overlap_chars": 50,
            },
        )

        self.assertEqual(["planner", "chunker", "synthesizer"], [step["agent"] for step in result["trace"]])
        self.assertIn("Synthesis was performed", result["memory"]["synthesizer"]["context"])


if __name__ == "__main__":
    unittest.main()
