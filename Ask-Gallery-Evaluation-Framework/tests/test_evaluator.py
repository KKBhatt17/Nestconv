import unittest

from evaluator import (
    EvaluationConfig,
    MetricSwitches,
    QueryGroundTruth,
    QueryPrediction,
    compute_rouge_l,
    compute_token_overlap_metrics,
    evaluate,
    normalize_text,
)


class EvaluatorTests(unittest.TestCase):
    def test_normalize_text(self) -> None:
        self.assertEqual(normalize_text("PNR: ABC-123,  Delhi"), "pnr abc 123 delhi")

    def test_token_metrics(self) -> None:
        scores = compute_token_overlap_metrics(
            prediction="terminal 2 gate 4",
            ground_truth="gate 4 terminal 2",
        )
        self.assertEqual(scores["token_precision"], 1.0)
        self.assertEqual(scores["token_recall"], 1.0)
        self.assertEqual(scores["token_f1"], 1.0)

    def test_rouge_l_penalizes_order_change(self) -> None:
        scores = compute_rouge_l(
            prediction="terminal 2 gate 4",
            ground_truth="gate 4 terminal 2",
        )
        self.assertLess(scores["rouge_l_f1"], 1.0)
        self.assertGreater(scores["rouge_l_f1"], 0.0)

    def test_end_to_end_evaluation(self) -> None:
        gt = {
            "what is my pnr?": QueryGroundTruth(
                query="what is my pnr?",
                image_answers=[("C:/docs/boarding_pass.png", "WYX567")],
            ),
            "what is my gate?": QueryGroundTruth(
                query="what is my gate?",
                image_answers=[("C:/docs/itinerary.png", "Gate 4")],
            ),
        }
        predictions = {
            "what is my pnr?": QueryPrediction(
                retrieved_images=["boarding_pass.png", "misc.png"],
                answer="The PNR is WYX567.",
            ),
            "what is my gate?": QueryPrediction(
                retrieved_images=["misc.png", "itinerary.png"],
                answer="terminal 2 gate 4",
            ),
        }
        config = EvaluationConfig(
            k_values=[1, 2],
            metrics=MetricSwitches(
                any_recall=True,
                all_recall=True,
                mrr=True,
                token_metrics=True,
                rouge_l=True,
                substring_inclusion=True,
                llm_judge=False,
            ),
        )

        results = evaluate(gt, predictions, config)

        self.assertEqual(results["summary"]["num_queries"], 2)
        self.assertEqual(results["summary"]["retrieval"]["by_k"]["k=1"]["any_recall"], 0.5)
        self.assertEqual(results["summary"]["retrieval"]["by_k"]["k=2"]["any_recall"], 1.0)
        self.assertAlmostEqual(results["summary"]["retrieval"]["mrr"], 0.75)
        self.assertGreater(results["summary"]["answering"]["token_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
