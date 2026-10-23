import unittest

import numpy as np
import pandas as pd

from streamlit_rr_dashboard.upload import parse_signal_csv, build_uploaded_replay
from streamlit_rr_dashboard.raw_features import _resp_features
from streamlit_rr_dashboard.core import source_breakdown
from streamlit_rr_dashboard.data import MODEL_FILE
import json


def csv_bytes(time_s, values, column):
    return pd.DataFrame({"time_s": time_s, column: values}).to_csv(index=False).encode()


class UploadTests(unittest.TestCase):
    def test_flat_respiratory_surrogate_is_not_a_candidate(self):
        rr, quality = _resp_features(np.zeros(128))
        self.assertTrue(np.isnan(rr))
        self.assertEqual(quality, 0.0)

    def test_parser_accepts_timestamped_ecg(self):
        t = np.arange(0, 33, .01)
        result = parse_signal_csv(csv_bytes(t, np.sin(t), "ecg"), "ecg")
        self.assertAlmostEqual(result.fs, 100.0)
        self.assertEqual(len(result.values), len(t))

    def test_parser_rejects_time_gap(self):
        t = np.r_[np.arange(0, 2, .01), np.arange(3, 5, .01)]
        with self.assertRaisesRegex(ValueError, "gap"):
            parse_signal_csv(csv_bytes(t, np.sin(t), "ecg"), "ecg")

    def test_parser_rejects_missing_samples(self):
        t = np.arange(0, 33, .01)
        values = np.sin(t)
        values[100] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_signal_csv(csv_bytes(t, values, "ppg"), "ppg")

    def test_uploaded_ecg_runs_moe_without_reference(self):
        fs = 100
        t = np.arange(0, 36, 1/fs)
        beat_phase = 2 * np.pi * 1.25 * t
        ecg = np.exp(-((np.sin(beat_phase / 2) / .13) ** 2)) * (1 + .12 * np.sin(2*np.pi*.25*t))
        signal = parse_signal_csv(csv_bytes(t, ecg, "ecg"), "ecg")
        replay = build_uploaded_replay(signal, None, reference_rr_bpm=None)
        self.assertGreaterEqual(len(replay), 1)
        self.assertIn("moe_rr_bpm", replay)
        self.assertTrue(replay.moe_rr_bpm.notna().any())
        self.assertTrue(replay.reference_rr_bpm.isna().all())
        self.assertTrue(replay.has_ecg.eq(1).all())
        self.assertTrue(replay.has_ppg.eq(0).all())

    def test_manual_reference_does_not_change_predictions(self):
        fs = 100
        t = np.arange(0, 33, 1/fs)
        ecg = np.sin(2*np.pi*1.2*t) + .1*np.sin(2*np.pi*.2*t)
        signal = parse_signal_csv(csv_bytes(t, ecg, "ecg"), "ecg")
        without = build_uploaded_replay(signal, None, None)
        with_reference = build_uploaded_replay(signal, None, 15.0)
        np.testing.assert_allclose(without.moe_rr_bpm, with_reference.moe_rr_bpm, equal_nan=True)
        self.assertTrue(with_reference.reference_rr_bpm.eq(15).all())

    def test_ppg_only_and_both_modalities(self):
        t = np.arange(0, 36, .01)
        ecg = np.exp(-((np.sin(np.pi*1.2*t)/.13)**2))*(1+.1*np.sin(2*np.pi*.25*t))
        ppg = np.sin(2*np.pi*1.2*t-.4)*(1+.1*np.sin(2*np.pi*.25*t))
        e = parse_signal_csv(csv_bytes(t, ecg, "ecg"), "ecg")
        p = parse_signal_csv(csv_bytes(t, ppg, "ppg"), "ppg")
        only_ppg = build_uploaded_replay(None, p, None)
        both = build_uploaded_replay(e, p, None)
        self.assertTrue(only_ppg.moe_rr_bpm.notna().any())
        self.assertTrue(only_ppg.has_ecg.eq(0).all())
        self.assertTrue(both.has_ecg.eq(1).all() and both.has_ppg.eq(1).all())
        candidates = json.loads(MODEL_FILE.read_text(encoding="utf-8"))["candidate_columns"]
        ppg_sources = source_breakdown(only_ppg.iloc[0], candidates).set_index("Source")
        self.assertEqual(ppg_sources.loc["ECG", "Gate share (%)"], 0.)
        self.assertEqual(ppg_sources.loc["Both", "Gate share (%)"], 0.)
        self.assertAlmostEqual(ppg_sources.loc["PPG", "Gate share (%)"], 100.)
        both_sources = source_breakdown(both.iloc[0], candidates).set_index("Source")
        self.assertAlmostEqual(both_sources["Gate share (%)"].sum(), 100.)


if __name__ == "__main__":
    unittest.main()
