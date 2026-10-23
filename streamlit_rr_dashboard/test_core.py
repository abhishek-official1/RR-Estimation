import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from streamlit_rr_dashboard.core import infer, source_breakdown, summarize_replay


MODEL = json.loads((Path(__file__).resolve().parents[1] / "moe_improvement/results/prototype_model.json").read_text())


class CoreTests(unittest.TestCase):
    def test_missing_experts_receive_zero_weight(self):
        row = {name: np.nan for name in MODEL["candidate_columns"] + MODEL["quality_columns"]}
        row[MODEL["candidate_columns"][0]] = 18.0
        row[MODEL["quality_columns"][0]] = 0.7
        result = infer(pd.DataFrame([row]), MODEL)
        self.assertEqual(result.loc[0, "selected_expert"], MODEL["candidate_columns"][0])
        self.assertAlmostEqual(result.loc[0, "moe_rr_bpm"], 18 + 3 * np.tanh(1.0))
        self.assertEqual(result.loc[0, "available_experts"], 1)

    def test_no_candidate_is_unavailable(self):
        row = {name: np.nan for name in MODEL["candidate_columns"] + MODEL["quality_columns"]}
        result = infer(pd.DataFrame([row]), MODEL)
        self.assertTrue(np.isnan(result.loc[0, "moe_rr_bpm"]))
        self.assertEqual(result.loc[0, "status"], "Unavailable")
        self.assertEqual(sum(result.loc[0, "expert_weights"]), 0.0)
        self.assertTrue(np.isnan(result.loc[0, "expert_disagreement_bpm"]))

    def test_expert_weights_and_disagreement_explain_prediction(self):
        model = {"candidate_columns": ["ecg_rr", "ppg_rr"],
                 "quality_columns": ["ecg_q", "ppg_q"],
                 "coefficients": [0., 0., 0., 0.]}
        frame = pd.DataFrame([{"ecg_rr": 10., "ppg_rr": 20., "ecg_q": .8, "ppg_q": .8,
                               "reference_rr_bpm": 40.}])
        result = infer(frame, model)
        self.assertEqual(result.loc[0, "expert_weights"], [0.5, 0.5])
        self.assertAlmostEqual(result.loc[0, "weighted_base_bpm"], 15.)
        self.assertAlmostEqual(result.loc[0, "expert_disagreement_bpm"], 5.)
        self.assertAlmostEqual(result.loc[0, "residual_bpm"], 0.)
        self.assertAlmostEqual(result.loc[0, "moe_rr_bpm"], 15.)

    def test_missing_expert_has_zero_weight_and_no_spread(self):
        model = {"candidate_columns": ["ecg_rr", "ppg_rr"],
                 "quality_columns": ["ecg_q", "ppg_q"],
                 "coefficients": [0., 0., 0., 0.]}
        result = infer(pd.DataFrame([{"ecg_rr": 12., "ppg_rr": np.nan,
                                      "ecg_q": .7, "ppg_q": .9}]), model)
        self.assertEqual(result.loc[0, "expert_weights"], [1.0, 0.0])
        self.assertAlmostEqual(result.loc[0, "expert_disagreement_bpm"], 0.)

    def test_replay_summary_reports_coverage_and_error_only_for_references(self):
        frame = pd.DataFrame({"moe_rr_bpm": [12., np.nan, 18.], "reference_rr_bpm": [10., 20., 20.]})
        summary = summarize_replay(frame)
        self.assertEqual(summary["coverage"], 2 / 3)
        self.assertEqual(summary["mae_bpm"], 2.0)

    def test_replay_summary_does_not_need_reference(self):
        summary = summarize_replay(pd.DataFrame({"moe_rr_bpm": [12., np.nan]}))
        self.assertEqual(summary["coverage"], 0.5)
        self.assertIsNone(summary["mae_bpm"])

    def test_source_breakdown_shows_gate_share_and_within_source_spread(self):
        candidates = ["ecg_amp_rr_consensus", "ecg_rsa_rr_consensus",
                      "ppg_amp_rr_consensus", "pred_all_quality_fusion"]
        row = pd.Series(dict(zip(candidates, [10., 20., 30., 24.])))
        row["expert_weights"] = [.25, .25, .25, .25]
        row["has_ecg"], row["has_ppg"] = 1, 1
        result = source_breakdown(row, candidates).set_index("Source")
        self.assertAlmostEqual(result.loc["ECG", "Gate share (%)"], 50.)
        self.assertAlmostEqual(result.loc["ECG", "Candidate RR (bpm)"], 15.)
        self.assertAlmostEqual(result.loc["ECG", "Spread (bpm)"], 5.)
        self.assertAlmostEqual(result.loc["PPG", "Gate share (%)"], 25.)
        self.assertTrue(np.isnan(result.loc["PPG", "Spread (bpm)"]))
        self.assertAlmostEqual(result.loc["Both", "Gate share (%)"], 25.)
        self.assertAlmostEqual(result["Gate share (%)"].sum(), 100.)

    def test_all_signal_candidate_is_attributed_to_present_modality(self):
        candidates = ["ecg_amp_rr_consensus", "pred_all_quality_fusion"]
        row = pd.Series({"ecg_amp_rr_consensus": 14., "pred_all_quality_fusion": 16.,
                         "expert_weights": [.4, .6], "has_ecg": 1, "has_ppg": 0})
        result = source_breakdown(row, candidates).set_index("Source")
        self.assertAlmostEqual(result.loc["ECG", "Gate share (%)"], 100.)
        self.assertEqual(result.loc["ECG", "Candidates"], 2)
        self.assertAlmostEqual(result.loc["ECG", "Spread (bpm)"], np.sqrt(.96))
        self.assertEqual(result.loc["Both", "Gate share (%)"], 0.)
        self.assertTrue(np.isnan(result.loc["Both", "Spread (bpm)"]))


if __name__ == "__main__":
    unittest.main()
