"""CPU-only NumPy inference for the notebook's saved single PhysioMoE.

The original checkpoint is a PyTorch state_dict. A restricted reader converts
its float32 tensors without importing PyTorch; the dashboard loads only the
packaged NPZ tensors produced by package_neural_moe.py.
"""

import pickle
import hashlib
import json
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ID_COLUMNS = {
    "dataset", "subject_id", "record_id", "reference_type", "start_sec", "window_sec",
    "reference_rr_bpm", "reference_quality", "reference_method", "reference_cycles",
    "reference_valid",
}


def _rebuild_tensor(storage, offset, shape, stride, requires_grad, hooks):
    if not isinstance(storage, np.ndarray) or len(shape) != len(stride):
        raise ValueError("Unexpected tensor storage")
    if shape and (offset + sum((n - 1) * s for n, s in zip(shape, stride)) >= len(storage)):
        raise ValueError("Tensor exceeds storage bounds")
    return np.ndarray(shape, dtype="<f4", buffer=storage, offset=offset * 4,
                      strides=tuple(s * 4 for s in stride)).copy()


class _FloatStorage:
    pass


def load_torch_state_dict(path: Path) -> dict[str, np.ndarray]:
    """Read only the known float32 tensor format of this local state_dict."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata = next((name for name in names if name.endswith("/data.pkl")), None)
        if metadata is None:
            raise ValueError("Checkpoint has no data.pkl")
        prefix = metadata.removesuffix("data.pkl")
        if prefix + "byteorder" in names and archive.read(prefix + "byteorder") != b"little":
            raise ValueError("Only little-endian checkpoint storage is supported")

        class RestrictedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                allowed = {
                    ("collections", "OrderedDict"): OrderedDict,
                    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
                    ("torch", "FloatStorage"): _FloatStorage,
                }
                if (module, name) not in allowed:
                    raise pickle.UnpicklingError(f"Disallowed checkpoint global: {module}.{name}")
                return allowed[(module, name)]

            def persistent_load(self, token):
                if (not isinstance(token, tuple) or len(token) != 5 or token[0] != "storage"
                        or token[1] is not _FloatStorage or token[3] != "cpu"):
                    raise pickle.UnpicklingError("Unsupported storage reference")
                key, count = str(token[2]), int(token[4])
                if not key.isdecimal() or count < 0:
                    raise pickle.UnpicklingError("Invalid storage key or size")
                raw = archive.read(prefix + "data/" + key)
                if len(raw) != count * 4:
                    raise ValueError("Storage size mismatch")
                return np.frombuffer(raw, dtype="<f4")

        import io
        loaded = RestrictedUnpickler(io.BytesIO(archive.read(metadata))).load()
    if not isinstance(loaded, OrderedDict) or not all(
        isinstance(k, str) and isinstance(v, np.ndarray) for k, v in loaded.items()
    ):
        raise ValueError("Checkpoint is not a tensor state_dict")
    return dict(loaded)


def fit_preprocessor(train: pd.DataFrame) -> dict:
    """Reproduce notebook FeaturePreprocessor.fit on its original training rows."""
    candidates = sorted(c for c in train if (c.endswith("_rr_consensus") or c.startswith("pred_"))
                        and c != "pred_adaptive_tracker")
    context = sorted(c for c in train if c not in ID_COLUMNS and c not in candidates
                     and pd.api.types.is_numeric_dtype(train[c]))
    candidates = [c for c in candidates if train[c].notna().any()]
    context = [c for c in context if train[c].notna().any()]
    if not candidates or not context:
        raise ValueError("Training rows have no usable neural MoE features")
    context_imputer = SimpleImputer(strategy="median", add_indicator=True)
    context_matrix = context_imputer.fit_transform(train[context])
    scaler = StandardScaler().fit(context_matrix)
    candidate_imputer = SimpleImputer(strategy="median").fit(train[candidates])
    return {
        "context_cols": context,
        "candidate_cols": candidates,
        "context_medians": context_imputer.statistics_.tolist(),
        "context_indicator_cols": context_imputer.indicator_.features_.tolist(),
        "context_mean": scaler.mean_.tolist(),
        "context_scale": scaler.scale_.tolist(),
        "candidate_medians": candidate_imputer.statistics_.tolist(),
        "training_rows": int(len(train)),
    }


def transform_features(frame: pd.DataFrame, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen notebook median imputation, indicators, and scaling."""
    missing = set(spec["context_cols"] + spec["candidate_cols"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing neural MoE columns: {sorted(missing)}")
    context = frame[spec["context_cols"]].to_numpy(dtype=float)
    indicators = np.isnan(context)[:, spec["context_indicator_cols"]].astype(float)
    context = np.where(np.isnan(context), np.asarray(spec["context_medians"]), context)
    context = np.column_stack((context, indicators)) if indicators.shape[1] else context
    context = (context - np.asarray(spec["context_mean"])) / np.asarray(spec["context_scale"])
    candidates = frame[spec["candidate_cols"]].to_numpy(dtype=float)
    candidates = np.where(np.isnan(candidates), np.asarray(spec["candidate_medians"]), candidates)
    candidates = np.clip(candidates, 3, 60)
    if not np.isfinite(context).all() or not np.isfinite(candidates).all():
        raise ValueError("Neural MoE received non-finite features")
    return context.astype(np.float32), candidates.astype(np.float32)


def _linear(x, weights, prefix):
    return x @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]


def _gelu(x):
    return .5 * x * (1 + erf(x / np.sqrt(2.0)))


def load_package(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Load the frozen numeric package without unpickling at dashboard runtime."""
    directory = Path(directory)
    spec = json.loads((directory / "preprocessor.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = directory.parents[1] / "Data/rr_uncertainty_exports/rr_uncertainty_cpu_results/moe_state.pt"
    if checkpoint.exists() and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["source_checkpoint_sha256"]:
        raise ValueError("Neural package does not match source checkpoint")
    with np.load(directory / "weights.npz", allow_pickle=False) as data:
        weights = {name: data[name] for name in data.files}
    if (len(spec["context_mean"]) != weights["context_net.0.weight"].shape[1]
            or len(spec["candidate_cols"]) != weights["gate.weight"].shape[0]):
        raise ValueError("Neural package dimensions are inconsistent")
    return spec, weights


def predict_numpy(frame: pd.DataFrame, spec: dict, weights: dict[str, np.ndarray]) -> pd.DataFrame:
    """Evaluate the frozen PhysioMoE architecture with dropout disabled."""
    context, candidates = transform_features(frame, spec)
    if (weights["context_net.0.weight"].shape[1] != context.shape[1]
            or weights["gate.weight"].shape[0] != candidates.shape[1]):
        raise ValueError("Preprocessor dimensions do not match neural checkpoint")
    first = _linear(context, weights, "context_net.0")
    mean = first.mean(axis=1, keepdims=True)
    variance = np.mean((first - mean) ** 2, axis=1, keepdims=True)
    norm = (first - mean) / np.sqrt(variance + 1e-5)
    norm = norm * weights["context_net.1.weight"] + weights["context_net.1.bias"]
    hidden = _gelu(_linear(_gelu(norm), weights, "context_net.4"))
    logits = _linear(hidden, weights, "gate")
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    gate = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    missing_candidates = np.isnan(frame[spec["candidate_cols"]].to_numpy(dtype=float))
    base = np.sum(gate * candidates, axis=1)
    residual = 3 * np.tanh(_linear(_gelu(_linear(hidden, weights, "residual.0")),
                                   weights, "residual.2").ravel() / 3)
    logvar = np.clip(_linear(_gelu(_linear(hidden, weights, "logvar.0")),
                             weights, "logvar.2").ravel(), -4, 5)
    output = pd.DataFrame(index=frame.index)
    output["neural_rr_bpm"] = base + residual
    output["neural_sd_aleatoric_bpm"] = np.sqrt(np.exp(logvar))
    output["neural_weighted_base_bpm"] = base
    output["neural_residual_bpm"] = residual
    output["neural_gate_weights"] = list(gate.tolist())
    output["neural_top_expert"] = np.asarray(spec["candidate_cols"])[gate.argmax(axis=1)]
    output["neural_observed_candidates"] = (~missing_candidates).sum(axis=1)
    output["neural_imputed_gate_weight"] = np.sum(gate * missing_candidates, axis=1)
    return output
