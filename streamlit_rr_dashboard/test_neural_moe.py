import unittest
import json

import numpy as np
import pandas as pd

from streamlit_rr_dashboard.neural_moe import (
    fit_preprocessor, load_package, load_torch_state_dict, predict_numpy, transform_features,
)
from streamlit_rr_dashboard.data import FEATURES, RESULTS, load_record, saved_recordings_available
from pathlib import Path


class NeuralMoETests(unittest.TestCase):
    def test_saved_checkpoint_contains_expected_layers(self):
        if not (RESULTS / "moe_state.pt").exists():
            self.skipTest("Original checkpoint is not in the public repository")
        weights = load_torch_state_dict(RESULTS / "moe_state.pt")
        self.assertEqual(weights["gate.weight"].shape[0], 14)
        self.assertEqual(weights["residual.2.weight"].shape[0], 1)
        self.assertEqual(weights["logvar.2.weight"].shape[0], 1)
        self.assertTrue(np.isfinite(weights["context_net.0.weight"]).all())

    def test_preprocessor_matches_training_imputation_and_indicators(self):
        training = pd.DataFrame({"dataset": ["a"]*3, "subject_id": ["1", "2", "3"],
                                 "context_a": [1., np.nan, 3.], "context_b": [5., 6., 7.],
                                 "ecg_amp_rr_consensus": [10., 20., np.nan]})
        spec = fit_preprocessor(training)
        context, candidates = transform_features(training, spec)
        self.assertEqual(spec["candidate_cols"], ["ecg_amp_rr_consensus"])
        self.assertEqual(context.shape, (3, 3))
        self.assertAlmostEqual(context[1, 2], (1 - 1/3) / np.std([0, 1, 0]), places=6)
        self.assertEqual(candidates[:, 0].tolist(), [10., 20., 15.])
        self.assertTrue(np.isfinite(context).all())

    def test_packaged_single_model_runs_saved_record_repeatably(self):
        if not saved_recordings_available():
            self.skipTest("Local study exports are not installed")
        package = Path(__file__).resolve().parent / "neural_package"
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["exact_preprocessor_reconstruction"])
        self.assertNotEqual(manifest["train_rows"], manifest["reported_train_rows"])
        spec, weights = load_package(package)
        frame = load_record(FEATURES, "bidmc", "bidmc_01_Signals",
                            ["record_id", "start_sec", "window_sec"] +
                            spec["context_cols"] + spec["candidate_cols"])
        first = predict_numpy(frame, spec, weights)
        second = predict_numpy(frame, spec, weights)
        self.assertEqual(len(first), len(frame))
        np.testing.assert_allclose(first.neural_rr_bpm, second.neural_rr_bpm, rtol=0, atol=0)
        self.assertTrue(first.neural_sd_aleatoric_bpm.gt(0).all())
        self.assertTrue(all(abs(sum(w)-1) < 1e-5 for w in first.neural_gate_weights))


if __name__ == "__main__":
    unittest.main()
