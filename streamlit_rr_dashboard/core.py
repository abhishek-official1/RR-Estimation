"""CPU inference and descriptive replay statistics for the saved masked MoE."""

import numpy as np
import pandas as pd


def infer(frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Predict from saved feature rows without using reference respiration."""
    candidates = model["candidate_columns"]
    quality_columns = model["quality_columns"]
    missing = set(candidates + quality_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    raw = frame[candidates].to_numpy(dtype=float)
    available = np.isfinite(raw) & (raw >= 3) & (raw <= 60)
    x = np.where(available, raw, 0.0)
    quality = np.clip(np.nan_to_num(frame[quality_columns].to_numpy(dtype=float), nan=0.0,
                                    posinf=0.0, neginf=0.0), 0.0, 1.0)
    theta = np.asarray(model["coefficients"], dtype=float)
    if theta.shape != (len(candidates) + 2,):
        raise ValueError("Model coefficient count does not match candidates")
    logits = theta[:len(candidates)][None, :] + theta[len(candidates)] * quality
    logits = np.where(available, logits, -1e6)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits) * available
    denominators = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, denominators, out=np.zeros_like(weights), where=denominators > 0)
    base = np.sum(weights * x, axis=1)
    residual = float(3 * np.tanh(theta[-1]))
    rr = np.clip(base + residual, 3, 60)
    has_candidate = available.any(axis=1)
    rr[~has_candidate] = np.nan
    spread = np.sqrt(np.sum(weights * (x - base[:, None]) ** 2, axis=1))
    spread[~has_candidate] = np.nan
    output = pd.DataFrame(index=frame.index)
    output["moe_rr_bpm"] = rr
    output["weighted_base_bpm"] = np.where(has_candidate, base, np.nan)
    output["residual_bpm"] = np.where(has_candidate, residual, np.nan)
    output["expert_disagreement_bpm"] = spread
    output["expert_weights"] = list(weights.tolist())
    output["available_experts"] = available.sum(axis=1)
    selected = np.asarray(candidates, dtype=object)[weights.argmax(axis=1)]
    output["selected_expert"] = np.where(has_candidate, selected, "")
    output["status"] = np.where(has_candidate, "Estimate · uncertainty unavailable", "Unavailable")
    return output


def source_breakdown(row: pd.Series, candidates: list[str]) -> pd.DataFrame:
    """Describe gate allocation and candidate disagreement by signal source.

    Candidate spread is descriptive and is not calibrated prediction uncertainty.
    The model's shared residual is excluded from source-specific candidate RR.
    """
    values = np.asarray([row[name] for name in candidates], dtype=float)
    weights = np.asarray(row["expert_weights"], dtype=float)
    if len(weights) != len(candidates):
        raise ValueError("Expert weights do not match the candidate list")
    valid = np.isfinite(values) & (values >= 3) & (values <= 60) & (weights > 0)
    ecg_active = any(valid[i] and (name.startswith("ecg_") or name.startswith("pred_ecg_"))
                     for i, name in enumerate(candidates))
    ppg_active = any(valid[i] and (name.startswith("ppg_") or name.startswith("pred_ppg_"))
                     for i, name in enumerate(candidates))
    if not ecg_active and not ppg_active:
        ecg_active, ppg_active = bool(row["has_ecg"]), bool(row["has_ppg"])

    groups = {"ECG": [], "PPG": [], "Both": []}
    for i, name in enumerate(candidates):
        if name.startswith(("ecg_", "pred_ecg_")):
            source = "ECG"
        elif name.startswith(("ppg_", "pred_ppg_")):
            source = "PPG"
        elif name.startswith("pred_all_"):
            source = "Both" if ecg_active and ppg_active else "ECG" if ecg_active else "PPG"
        else:
            raise ValueError(f"Unknown RR candidate source: {name}")
        if valid[i]:
            groups[source].append(i)

    rows = []
    for source, indices in groups.items():
        share = float(weights[indices].sum()) if indices else 0.0
        estimate = float(np.dot(weights[indices], values[indices]) / share) if share > 0 else np.nan
        spread = (float(np.sqrt(np.dot(weights[indices], (values[indices] - estimate) ** 2) / share))
                  if len(indices) >= 2 and share > 0 else np.nan)
        rows.append({"Source": source, "Gate share (%)": share * 100,
                     "Candidate RR (bpm)": estimate, "Spread (bpm)": spread,
                     "Candidates": len(indices)})
    return pd.DataFrame(rows)


def summarize_replay(frame: pd.DataFrame) -> dict:
    """Report retained coverage and optional descriptive MAE for one replay."""
    retained = np.isfinite(frame["moe_rr_bpm"].to_numpy(dtype=float))
    coverage = float(retained.mean()) if len(frame) else 0.0
    mae = None
    if "reference_rr_bpm" in frame:
        reference = frame["reference_rr_bpm"].to_numpy(dtype=float)
        valid = retained & np.isfinite(reference)
        if valid.any():
            mae = float(np.abs(frame["moe_rr_bpm"].to_numpy(dtype=float)[valid] - reference[valid]).mean())
    return {"windows": len(frame), "retained": int(retained.sum()), "coverage": coverage, "mae_bpm": mae}
