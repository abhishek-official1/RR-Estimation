"""Validate timestamped CSV recordings and run a bounded CPU MoE replay."""

import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from streamlit_rr_dashboard.core import infer
from streamlit_rr_dashboard.data import MODEL_FILE
from streamlit_rr_dashboard.raw_features import extract_window


MAX_BYTES = 20_000_000
MAX_SECONDS = 300
WINDOW_SECONDS = 32
STRIDE_SECONDS = 4


@dataclass(frozen=True)
class UploadedSignal:
    time_s: np.ndarray
    values: np.ndarray
    fs: float
    kind: str


def parse_signal_csv(content: bytes, kind: str) -> UploadedSignal:
    """Require time_s plus ecg/ppg columns with finite, nearly uniform samples."""
    if kind not in ("ecg", "ppg"):
        raise ValueError("Signal kind must be ecg or ppg")
    if not content or len(content) > MAX_BYTES:
        raise ValueError("CSV must be nonempty and no larger than 20 MB")
    try:
        frame = pd.read_csv(io.BytesIO(content), usecols=["time_s", kind])
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"{kind.upper()} CSV needs columns time_s and {kind}") from exc
    if len(frame) < 32:
        raise ValueError(f"{kind.upper()} CSV has too few samples")
    try:
        times = pd.to_numeric(frame.time_s, errors="raise").to_numpy(dtype=float)
        values = pd.to_numeric(frame[kind], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{kind.upper()} CSV needs numeric time and signal values") from exc
    if not (np.isfinite(times).all() and np.isfinite(values).all()):
        raise ValueError(f"{kind.upper()} CSV needs finite time and signal values")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError(f"{kind.upper()} time_s must increase strictly")
    interval = float(np.median(dt))
    fs = 1/interval
    if not 20 <= fs <= 1000:
        raise ValueError(f"{kind.upper()} sampling rate must be 20–1000 Hz")
    if np.max(np.abs(dt-interval)) > max(.1*interval, .0001):
        raise ValueError(f"{kind.upper()} CSV has a sampling gap or irregular timestamps")
    duration = times[-1]-times[0]+interval
    if duration > MAX_SECONDS:
        raise ValueError(f"{kind.upper()} recording exceeds {MAX_SECONDS} seconds")
    if duration < WINDOW_SECONDS:
        raise ValueError(f"{kind.upper()} recording needs at least 32 seconds")
    if np.std(values) < 1e-10:
        raise ValueError(f"{kind.upper()} signal is flat")
    return UploadedSignal(times, values, fs, kind)


def _window(recording: UploadedSignal | None, start: float):
    if recording is None:
        return None
    end = start + WINDOW_SECONDS
    if start < recording.time_s[0] or end > recording.time_s[-1] + 1/recording.fs:
        return None
    first = int(np.searchsorted(recording.time_s, start, side="left"))
    last = int(np.searchsorted(recording.time_s, end, side="left"))
    samples = recording.values[first:last]
    expected = int(round(WINDOW_SECONDS*recording.fs))
    return samples if len(samples) >= expected-2 else None


def build_uploaded_replay(ecg: UploadedSignal | None, ppg: UploadedSignal | None,
                          reference_rr_bpm: float | None) -> pd.DataFrame:
    """Extract 32-second windows, run the saved masked MoE, and attach optional reference."""
    if ecg is None and ppg is None:
        raise ValueError("Upload ECG, PPG, or both")
    if reference_rr_bpm is not None and not 3 <= reference_rr_bpm <= 60:
        raise ValueError("Reference RR must be between 3 and 60 breaths/min")
    recordings = [x for x in (ecg, ppg) if x is not None]
    start = min(x.time_s[0] for x in recordings)
    stop = max(x.time_s[-1] + 1/x.fs for x in recordings)
    rows = []
    for offset in np.arange(0, stop-start-WINDOW_SECONDS+1e-6, STRIDE_SECONDS):
        window_start = float(start+offset)
        ecg_window = _window(ecg, window_start)
        ppg_window = _window(ppg, window_start)
        if ecg_window is None and ppg_window is None:
            continue
        row = extract_window(ecg_window, ecg.fs if ecg_window is not None else None,
                             ppg_window, ppg.fs if ppg_window is not None else None)
        row.update({"record_id": "uploaded_recording", "start_sec": window_start,
                    "window_sec": WINDOW_SECONDS, "reference_rr_bpm": reference_rr_bpm})
        rows.append(row)
    if not rows:
        raise ValueError("No complete 32-second ECG or PPG windows overlap the upload")
    frame = pd.DataFrame(rows)
    model = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    for name in model["candidate_columns"] + model["quality_columns"] + ["ecg_sqi_composite", "ppg_sqi_composite"]:
        if name not in frame:
            frame[name] = np.nan
    if reference_rr_bpm is None:
        frame["reference_rr_bpm"] = np.nan
    result = infer(frame, model)
    return pd.concat([frame, result], axis=1)
