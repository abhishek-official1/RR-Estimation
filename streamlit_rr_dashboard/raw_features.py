"""Exploratory CPU feature extraction for uploaded ECG/PPG CSV signals.

This follows the notebook's surrogate families and summary columns. Peak detection
and AR estimation differ because NeuroKit2 and statsmodels are not installed.
"""

import numpy as np
import warnings
from scipy import signal


PATHWAYS = ("ecg_amp", "ecg_rsa", "ecg_baseline", "ecg_morph",
            "ppg_interval", "ppg_amp", "ppg_baseline", "ppg_width")
RESP_BAND = (3/60, 1.0)
GRID_FS = 4.0


def _filter(x, fs, band):
    hi = min(band[1], .45 * fs)
    if band[0] >= hi:
        return np.asarray(x, float)
    sos = signal.butter(4, (band[0], hi), btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def _norm(x):
    x = np.asarray(x, float)
    med = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - med))
    if scale < 1e-8:
        scale = np.std(x)
    return (x - med) / scale if scale >= 1e-8 else np.zeros_like(x)


def _grid(values, times, duration=32):
    if len(values) < 4 or not np.isfinite(values).all():
        return None
    grid = np.arange(0, duration, 1/GRID_FS)
    return _norm(np.interp(grid, times, values))


def _peaks(x, fs, kind):
    if kind == "ecg":
        f = _filter(x, fs, (5, 25))
        peaks, _ = signal.find_peaks(np.abs(f), distance=max(1, int(fs*.25)),
                                     prominence=max(np.std(f)*.4, 1e-8))
    else:
        f = _filter(x, fs, (.5, 5))
        peaks, _ = signal.find_peaks(f, distance=max(1, int(fs*.25)),
                                     prominence=max(np.std(f)*.25, 1e-8))
    return peaks


def _spectral(x):
    x = _norm(x)
    f, p = signal.welch(x, fs=GRID_FS, nperseg=min(len(x), 64), detrend="linear")
    mask = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    fb, pb = f[mask], p[mask]
    if not len(pb) or np.sum(pb) <= 0:
        return np.nan, 0.0, 1.0
    i = np.argmax(pb)
    peak_f = fb[i]
    local = np.abs(fb-peak_f) <= max(.03, 1/(len(x)/GRID_FS))
    concentration = np.sum(pb[local])/(np.sum(pb)+1e-12)
    q = pb/np.sum(pb)
    entropy = -np.sum(q*np.log(q+1e-12))/np.log(len(q)) if len(q) > 1 else 1.0
    return float(peak_f), float(concentration), float(entropy)


def _resp_features(x):
    if len(x) < 16 or not np.isfinite(x).all() or np.std(x) < 1e-8:
        return np.nan, 0.0
    f, concentration, entropy = _spectral(x)
    ac = signal.correlate(_norm(x), _norm(x), mode="full", method="fft")[len(x)-1:]
    lo = max(1, int(GRID_FS*60/60))
    hi = min(len(ac)-1, int(GRID_FS*60/3))
    acq = float(np.clip(np.max(ac[lo:hi+1])/ac[0], 0, 1)) if hi >= lo and ac[0] > 0 else 0.0
    rqi = float(np.clip(np.mean((concentration, acq, 1-entropy)), 0, 1))
    n = max(256, 2**int(np.ceil(np.log2(len(x)))))
    fft_f = np.fft.rfftfreq(n, d=1/GRID_FS)
    power = np.abs(np.fft.rfft(_norm(x), n=n))**2
    band = (fft_f >= RESP_BAND[0]) & (fft_f <= RESP_BAND[1])
    fft_rr = 60*fft_f[band][np.argmax(power[band])] if band.any() else np.nan
    ac_peaks, _ = signal.find_peaks(ac[lo:hi+1]) if hi >= lo else ([], {})
    lag = lo + (ac_peaks[np.argmax(ac[lo:hi+1][ac_peaks])] if len(ac_peaks) else np.argmax(ac[lo:hi+1])) if hi >= lo else 0
    ac_rr = 60*GRID_FS/lag if lag > 0 else np.nan
    peaks, _ = signal.find_peaks(_norm(x), distance=max(1,int(GRID_FS*.65)), prominence=.2)
    periods = np.diff(peaks)/GRID_FS
    periods = periods[(periods >= 1) & (periods <= 20)]
    peak_rr = 60/np.median(periods) if len(periods) >= 2 else np.nan
    estimates = np.array((fft_rr, 60*f if np.isfinite(f) else np.nan, ac_rr, peak_rr), float)
    return float(np.median(estimates[np.isfinite(estimates)])) if np.isfinite(estimates).any() else np.nan, rqi


def _morphology(ecg_f, peaks, fs):
    beats, times = [], []
    for p in peaks:
        i0, i1 = int(p-.25*fs), int(p+.45*fs)
        if i0 < 0 or i1 > len(ecg_f) or i1-i0 < 8:
            continue
        beat = np.interp(np.linspace(0,1,100), np.linspace(0,1,i1-i0), ecg_f[i0:i1])
        beats.append(_norm(beat))
        times.append(p/fs)
    if len(beats) < 4:
        return None
    matrix = np.vstack(beats)
    u, s, _ = np.linalg.svd(matrix-matrix.mean(axis=0), full_matrices=False)
    return _grid(u[:,0]*s[0], np.asarray(times))


def _surrogates_ecg(x, fs):
    filtered = _filter(x, fs, (.5, 40))
    peaks = _peaks(filtered, fs, "ecg")
    out = {}
    if len(peaks) < 4:
        return out, peaks
    times = peaks/fs
    out["ecg_amp"] = _grid(filtered[peaks], times)
    intervals = np.diff(peaks)/fs
    valid = (intervals >= .25) & (intervals <= 2.4)
    if valid.sum() >= 4:
        out["ecg_rsa"] = _grid(intervals[valid], (peaks[1:][valid]+peaks[:-1][valid])/(2*fs))
    base = signal.sosfiltfilt(signal.butter(3, 1.15, btype="lowpass", fs=fs, output="sos"), x)
    out["ecg_baseline"] = _grid(base[peaks], times)
    out["ecg_morph"] = _morphology(filtered, peaks, fs)
    return {k:v for k,v in out.items() if v is not None}, peaks


def _surrogates_ppg(x, fs):
    filtered = _filter(x, fs, (.3, 8))
    peaks = _peaks(filtered, fs, "ppg")
    out = {}
    if len(peaks) < 4:
        return out, peaks
    times = peaks/fs
    intervals = np.diff(peaks)/fs
    valid = (intervals >= .25) & (intervals <= 2.4)
    if valid.sum() >= 4:
        out["ppg_interval"] = _grid(intervals[valid], (peaks[1:][valid]+peaks[:-1][valid])/(2*fs))
    troughs, _ = signal.find_peaks(-filtered, distance=max(1,int(fs*.2)))
    amps, bases, amp_times = [], [], []
    for p in peaks:
        before = troughs[troughs < p]
        if len(before):
            t = before[-1]
            amps.append(filtered[p]-filtered[t])
            bases.append(filtered[t])
            amp_times.append(p/fs)
    out["ppg_amp"] = _grid(amps, amp_times)
    out["ppg_baseline"] = _grid(bases, amp_times)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="some peaks have a (prominence|width) of 0")
        widths = signal.peak_widths(filtered, peaks, rel_height=.5)[0]/fs
    good = np.isfinite(widths) & (widths > 0) & (widths < 2)
    out["ppg_width"] = _grid(widths[good], times[good])
    return {k:v for k,v in out.items() if v is not None}, peaks


def _weighted(values, qualities):
    keys = [k for k,v in values.items() if np.isfinite(v) and 3 <= v <= 60 and qualities[k] > 0]
    if not keys:
        return np.nan, 0.0
    v = np.array([values[k] for k in keys])
    w = np.array([qualities[k] for k in keys])
    weights = w/w.sum()
    pred = np.sum(weights*v)
    confidence = np.mean(w)*np.exp(-np.sqrt(np.sum(weights*(v-pred)**2))/3)
    return float(pred), float(np.clip(confidence,0,1))


def _pca_rr(surrogates):
    if len(surrogates) < 2:
        return np.nan
    matrix = np.column_stack([_norm(x) for x in surrogates.values()])
    u, s, _ = np.linalg.svd(matrix-matrix.mean(axis=0), full_matrices=False)
    f, _, _ = _spectral(u[:,0]*s[0])
    return 60*f if np.isfinite(f) else np.nan


def extract_window(ecg=None, ecg_fs=None, ppg=None, ppg_fs=None):
    """Return the compact MoE's candidate and quality columns for one window."""
    row = {"has_ecg": int(ecg is not None), "has_ppg": int(ppg is not None)}
    surrogates = {}
    if ecg is not None:
        ss, peaks = _surrogates_ecg(np.asarray(ecg,float), ecg_fs)
        surrogates.update(ss)
        row["ecg_sqi_composite"] = float(np.mean((min(1,len(peaks)/20), 1-np.mean(np.asarray(ecg)==np.min(ecg)))))
    if ppg is not None:
        ss, peaks = _surrogates_ppg(np.asarray(ppg,float), ppg_fs)
        surrogates.update(ss)
        row["ppg_sqi_composite"] = float(np.mean((min(1,len(peaks)/20), 1-np.mean(np.asarray(ppg)==np.min(ppg)))))
    values, qualities = {}, {}
    for name, x in surrogates.items():
        rr, quality = _resp_features(x)
        row[f"{name}_rr_consensus"] = rr
        row[f"{name}_rqi_overall"] = quality
        values[name], qualities[name] = rr, quality
    for prefix, keys in (("ecg", [k for k in values if k.startswith("ecg_")]),
                         ("ppg", [k for k in values if k.startswith("ppg_")]),
                         ("all", list(values))):
        pred, conf = _weighted({k:values[k] for k in keys}, {k:qualities[k] for k in keys})
        row[f"pred_{prefix}_quality_fusion"] = pred
        row[f"conf_{prefix}_quality_fusion"] = conf
        row[f"pred_{prefix}_pca"] = _pca_rr({k:surrogates[k] for k in keys})
    row["rqi_composite"] = float(np.mean(list(qualities.values()))) if qualities else np.nan
    return row
