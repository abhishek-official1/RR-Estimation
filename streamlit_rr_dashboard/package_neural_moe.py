"""Package the saved single PhysioMoE for CPU-only NumPy inference.

Run once from the project root. Reads cached 32-second features; never trains.
"""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from streamlit_rr_dashboard.data import FEATURES, RESULTS
from streamlit_rr_dashboard.neural_moe import fit_preprocessor, load_torch_state_dict, predict_numpy


OUT = Path(__file__).resolve().parent / "neural_package"
SEED = 20261003
TIME_LIMIT_SECONDS = 180


def main():
    start = time.monotonic()
    files = sorted(FEATURES.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No saved 32-second feature tables")
    features = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True, sort=False)
    valid = features.reference_rr_bpm.between(3, 60, inclusive="both") & features.reference_quality.gt(0)
    features = features.loc[valid & features.dataset.ne("capnobase")].copy()
    if features.empty:
        raise ValueError("No valid development feature rows")

    # Exact notebook procedure: original subject IDs and 15%+15% group holdout.
    groups = features.subject_id.to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=SEED)
    train_idx, _ = next(splitter.split(features, groups=groups))
    train = features.iloc[train_idx]
    spec = fit_preprocessor(train)
    weights_path = RESULTS / "moe_state.pt"
    weights = load_torch_state_dict(weights_path)
    if (len(spec["context_mean"]) != weights["context_net.0.weight"].shape[1]
            or len(spec["candidate_cols"]) != weights["gate.weight"].shape[0]):
        raise ValueError("Reconstructed preprocessing does not match saved neural checkpoint")
    sample = predict_numpy(train.head(8), spec, weights)
    if not np.isfinite(sample.neural_rr_bpm).all():
        raise ValueError("Packaged neural MoE produced non-finite predictions")
    elapsed = time.monotonic() - start
    if elapsed > TIME_LIMIT_SECONDS:
        raise TimeoutError(f"Packaging exceeded {TIME_LIMIT_SECONDS} seconds")

    reported = pd.read_csv(RESULTS / "tables/stages/subject_independent_splits.csv")
    reported_train_rows = int(reported.loc[reported.split.eq("train"), "windows"].iloc[0])
    exact_preprocessor_reconstruction = len(train) == reported_train_rows

    OUT.mkdir(exist_ok=True)
    (OUT / "preprocessor.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "weights.npz", **weights)
    manifest = {
        "model": "original single PhysioMoE checkpoint",
        "source_checkpoint_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "split_seed": SEED,
        "train_rows": len(train),
        "reported_train_rows": reported_train_rows,
        "exact_preprocessor_reconstruction": exact_preprocessor_reconstruction,
        "train_subject_ids": int(train.subject_id.nunique()),
        "development_rows": len(features),
        "context_dimensions": len(spec["context_mean"]),
        "candidate_dimensions": len(spec["candidate_cols"]),
        "sample_predictions_bpm": sample.neural_rr_bpm.round(6).tolist(),
        "seconds": elapsed,
        "notes": "The saved training split does not match the split reproduced from available "
                 "feature tables. Original fitted imputer/scaler parameters were not saved; "
                 "predictions are experimental and are not a validated reproduction. "
                 "Single checkpoint only; saved ensemble member weights are unavailable.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
