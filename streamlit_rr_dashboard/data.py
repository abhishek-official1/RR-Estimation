"""Read only the chosen record from the saved window feature tables."""

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Data/rr_uncertainty_exports/rr_uncertainty_cpu_results"
FEATURES = RESULTS / "features/window_32s"
INVENTORY = RESULTS / "record_inventory.parquet"
MODEL_FILE = ROOT / "moe_improvement/results/prototype_model.json"
DISPLAY_COLUMNS = ["record_id", "start_sec", "window_sec", "reference_rr_bpm",
                   "has_ecg", "has_ppg", "ecg_sqi_composite", "ppg_sqi_composite",
                   "rqi_composite", "pred_all_quality_fusion"]


def saved_recordings_available(inventory: Path = INVENTORY, feature_root: Path = FEATURES) -> bool:
    """Enable saved replay only where the local study exports are installed."""
    return inventory.is_file() and feature_root.is_dir() and any(feature_root.rglob("*.parquet"))


def record_catalog() -> pd.DataFrame:
    if not INVENTORY.exists():
        raise FileNotFoundError(f"Missing saved record inventory: {INVENTORY}")
    catalog = pd.read_parquet(INVENTORY, columns=["dataset", "subject_id", "record_id", "windows"])
    return catalog.loc[catalog.windows > 0].sort_values(["dataset", "record_id"]).reset_index(drop=True)


def load_record(feature_root: Path, dataset: str, record_id: str, columns: list[str]) -> pd.DataFrame:
    """Load one record, selecting columns and filtering in Arrow before conversion."""
    folder = feature_root / dataset
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing feature dataset: {folder}")
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature parquet files in {folder}")
    source = ds.dataset(files, format="parquet")
    available = [name for name in columns if name in source.schema.names]
    required = {"record_id", "start_sec", "window_sec"}
    if not required.issubset(available):
        raise ValueError(f"Missing required feature columns: {sorted(required - set(available))}")
    table = source.to_table(columns=available, filter=ds.field("record_id") == record_id, use_threads=False)
    frame = table.to_pandas().sort_values("start_sec").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No saved windows for {dataset}/{record_id}")
    for name in columns:
        if name not in frame:
            frame[name] = float("nan")
    return frame
