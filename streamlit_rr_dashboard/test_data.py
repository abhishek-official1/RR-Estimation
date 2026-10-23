import unittest

from streamlit_rr_dashboard.data import FEATURES, INVENTORY, load_record, saved_recordings_available


class DataTests(unittest.TestCase):
    def test_saved_mode_requires_inventory_and_feature_files(self):
        self.assertFalse(saved_recordings_available(INVENTORY.with_name("absent.parquet"), FEATURES))
        self.assertFalse(saved_recordings_available(INVENTORY, FEATURES / "absent"))
        if INVENTORY.exists() and FEATURES.is_dir():
            self.assertTrue(saved_recordings_available(INVENTORY, FEATURES))

    def test_load_record_filters_and_orders_windows(self):
        if not saved_recordings_available():
            self.skipTest("Local study exports are not installed")
        result = load_record(FEATURES, "bidmc", "bidmc_01_Signals",
                             ["record_id", "start_sec", "window_sec", "reference_rr_bpm"])
        self.assertGreater(len(result), 0)
        self.assertEqual(result.start_sec.tolist(), sorted(result.start_sec.tolist()))
        self.assertEqual(result.record_id.unique().tolist(), ["bidmc_01_Signals"])

    def test_missing_modality_columns_are_nan(self):
        if not saved_recordings_available():
            self.skipTest("Local study exports are not installed")
        result = load_record(FEATURES, "cebsdb", "b001", ["record_id", "start_sec", "window_sec", "ppg_amp_rr_consensus"])
        self.assertIn("ppg_amp_rr_consensus", result)
        self.assertTrue(result.ppg_amp_rr_consensus.isna().all())


if __name__ == "__main__":
    unittest.main()
