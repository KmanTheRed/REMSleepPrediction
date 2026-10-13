import unittest

import evaluate


class PostprocessingTests(unittest.TestCase):
    def test_merge_then_floor(self):
        issues = {
            "prediction": {
                "clipped_intervals": 0,
                "discarded_nonpositive_intervals": 0,
                "discarded_short_after_merge": 0,
            }
        }
        result = evaluate.normalize_intervals(
            [
                {"start_sec": 10, "end_sec": 27},
                {"start_sec": 32, "end_sec": 48},
                {"start_sec": 100, "end_sec": 129},
            ],
            issues,
            "test",
        )
        self.assertEqual(result, [(10.0, 48.0)])
        self.assertEqual(issues["prediction"]["discarded_short_after_merge"], 1)

    def test_exact_five_second_gap_merges(self):
        issues = {
            "prediction": {
                "clipped_intervals": 0,
                "discarded_nonpositive_intervals": 0,
                "discarded_short_after_merge": 0,
            }
        }
        result = evaluate.normalize_intervals(
            [{"start_sec": 0, "end_sec": 30}, {"start_sec": 35, "end_sec": 65}],
            issues,
            "test",
        )
        self.assertEqual(result, [(0.0, 65.0)])


class MatchingAndMetricsTests(unittest.TestCase):
    def test_one_prediction_cannot_match_two_truth_events(self):
        truth = [(0.0, 40.0), (45.0, 85.0)]
        prediction = [(0.0, 85.0)]
        matches = evaluate.ordered_one_to_one_matches(truth, prediction, 0.3)
        self.assertEqual(len(matches), 1)

    def test_perfect_score(self):
        frames = [{"frame_id": "F_x", "truth": [(100.0, 150.0)], "prediction": [(100.0, 150.0)]}]
        report = evaluate.score_frames(frames)
        self.assertEqual(report["per_second"]["f1"], 1.0)
        self.assertEqual(report["per_second"]["cohens_kappa"], 1.0)
        for metric in report["events_by_iou"].values():
            self.assertEqual(metric["f1"], 1.0)

    def test_empty_prediction(self):
        frames = [{"frame_id": "F_x", "truth": [(100.0, 150.0)], "prediction": []}]
        report = evaluate.score_frames(frames)
        self.assertEqual(report["per_second"]["f1"], 0.0)
        self.assertEqual(report["events_by_iou"]["0.3"]["recall"], 0.0)

    def test_low_coverage_frame_is_required_but_excluded(self):
        vault = [
            {
                "frame_id": "F_x",
                "stage": "test_01",
                "raw_episodes": [],
                "render": {"eligible_for_scoring": False},
            }
        ]
        predictions = [{"frame_id": "F_x", "episodes": []}]
        by_stage, issues = evaluate.prepare(vault, predictions, ["test_01"])
        self.assertEqual(by_stage["test_01"], [])
        self.assertEqual(issues["quality_exclusions"]["test_01"], 1)


if __name__ == "__main__":
    unittest.main()
