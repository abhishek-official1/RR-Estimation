"""Streamlit research replay of saved 32-second ECG/PPG feature windows."""

import json
import hashlib
from html import escape
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Allow `streamlit run streamlit_rr_dashboard/app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit_rr_dashboard.core import infer, source_breakdown, summarize_replay
from streamlit_rr_dashboard.data import (
    DISPLAY_COLUMNS, FEATURES, MODEL_FILE, load_record, record_catalog, saved_recordings_available,
)
from streamlit_rr_dashboard.neural_moe import load_package, predict_numpy
from streamlit_rr_dashboard.upload import parse_signal_csv, build_uploaded_replay

NEURAL_PACKAGE = Path(__file__).with_name("neural_package")


st.set_page_config(page_title="Respira / RR Research Monitor", page_icon="◉", layout="wide")
st.markdown(f"<style>{(Path(__file__).with_name('theme.css')).read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)


def fact_card(label: str, value: str, hint: str = ""):
    st.markdown(
        f'<div class="fact-card"><span class="label">{escape(label)}</span>'
        f'<strong class="value">{escape(value)}</strong>'
        f'<span class="hint">{escape(hint)}</span></div>', unsafe_allow_html=True)


def mini_stat(label: str, value: str):
    st.markdown(f'<div class="mini-stat"><div class="label">{escape(label)}</div>'
                f'<div class="value">{escape(value)}</div></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_catalog():
    return record_catalog()


@st.cache_data(show_spinner="Loading saved feature windows…")
def get_replay(dataset: str, record_id: str, model_text: str):
    model = json.loads(model_text)
    neural_spec, neural_weights = get_neural_package()
    columns = list(dict.fromkeys(DISPLAY_COLUMNS + model["candidate_columns"] + model["quality_columns"]
                                 + neural_spec["context_cols"] + neural_spec["candidate_cols"]))
    frame = load_record(FEATURES, dataset, record_id, columns)
    masked_result = infer(frame, model)
    neural_result = predict_numpy(frame, neural_spec, neural_weights)
    return pd.concat([frame, masked_result, neural_result], axis=1)


@st.cache_resource(show_spinner=False)
def get_neural_package():
    return load_package(NEURAL_PACKAGE)


st.markdown('<div class="brandbar"><div class="brand"><span class="brandmark" aria-hidden="true"></span>'
            'Respira <span class="brand-sub">/ Research monitor</span></div>'
            '<div class="brandmeta">Local inference · CPU only</div></div>', unsafe_allow_html=True)
st.markdown('<section class="hero"><div><div class="eyebrow">Respiratory research / live replay</div>'
            '<h1>Every breath,<br><em>in view.</em></h1></div>'
            '<div class="hero-visual" aria-hidden="true"><span class="trace-label">ECG / PPG → RR</span>'
            '<svg viewBox="0 0 480 120" preserveAspectRatio="none"><path '
            'd="M0 64 H72 L87 63 L97 51 L105 83 L114 15 L122 106 L133 62 H185 '
            'Q206 62 222 59 Q240 51 253 62 H315 L331 62 L340 49 L348 83 L357 16 '
            'L365 103 L375 63 H480"/></svg><span class="trace-end">32 S WINDOW</span>'
            '</div></section>', unsafe_allow_html=True)

try:
    model_text = MODEL_FILE.read_text(encoding="utf-8")
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

saved_available = saved_recordings_available()
with st.sidebar:
    st.markdown('<div class="eyebrow">01 / INPUT</div>', unsafe_allow_html=True)
    st.header("Choose a source")
    sources = ["Saved recording", "Upload CSV"] if saved_available else ["Upload CSV"]
    source = st.radio("Input", sources, horizontal=True)
    if source == "Saved recording":
        catalog = get_catalog()
        datasets = sorted(catalog.dataset.unique())
        dataset = st.selectbox("Dataset", datasets, index=datasets.index("bidmc") if "bidmc" in datasets else 0)
        subset = catalog.loc[catalog.dataset.eq(dataset)]
        records = subset.record_id.astype(str).tolist()
        record_id = st.selectbox("Recording", records)
        show_reference = st.checkbox("Show reference RR and descriptive error", value=False)
        model_choice = st.selectbox("Primary model", ["Neural checkpoint (experimental)", "Masked MoE"])
    else:
        dataset, record_id = "upload", "uploaded_recording"
        st.caption("CSV: `time_s` + `ecg` or `ppg`.")
        ecg_file = st.file_uploader("ECG CSV", type="csv", key="ecg_csv")
        ppg_file = st.file_uploader("PPG CSV", type="csv", key="ppg_csv")
        use_reference = st.checkbox("I have a real/reference RR value")
        reference_rr = st.number_input("Real RR (breaths/min)", min_value=3.0, max_value=60.0,
                                       value=15.0, step=0.1, disabled=not use_reference)
        show_reference = use_reference
        model_choice = "Masked MoE"
        run_upload = st.button("Run MoE on upload", type="primary", width="stretch")
    with st.expander("Input & model notes"):
        st.caption("Reference RR is for comparison only; it does not enter inference.")
        if not saved_available:
            st.caption("This public app processes uploads on a cloud server. Use de-identified or test recordings.")
        if source == "Saved recording":
            st.caption("The neural checkpoint uses reconstructed preprocessing. Its original training split "
                       "could not be reproduced. Aleatoric SD is not calibrated confidence.")
        else:
            st.caption("Uploads use the masked MoE. CSV feature extraction has not been validated "
                       "for the neural checkpoint.")

if source == "Saved recording":
    try:
        replay = get_replay(dataset, record_id, model_text)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(f"Cannot load this recording: {exc}")
        st.stop()
    replay_key = f"{dataset}|{record_id}|{model_choice}"
else:
    ecg_bytes = ecg_file.getvalue() if ecg_file else None
    ppg_bytes = ppg_file.getvalue() if ppg_file else None
    fingerprint = hashlib.sha256((ecg_bytes or b"") + b"|PPG|" + (ppg_bytes or b"") +
                                 str(reference_rr if use_reference else None).encode()).hexdigest()
    if run_upload:
        try:
            ecg_signal = parse_signal_csv(ecg_bytes, "ecg") if ecg_bytes else None
            ppg_signal = parse_signal_csv(ppg_bytes, "ppg") if ppg_bytes else None
            with st.spinner("Extracting respiratory features and running the MoE on CPU…"):
                st.session_state.uploaded_replay = build_uploaded_replay(
                    ecg_signal, ppg_signal, float(reference_rr) if use_reference else None)
            st.session_state.uploaded_ecg = ecg_signal
            st.session_state.uploaded_ppg = ppg_signal
            st.session_state.uploaded_fingerprint = fingerprint
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
            st.stop()
    if st.session_state.get("uploaded_fingerprint") != fingerprint:
        st.info("Upload ECG and/or PPG CSV files, then select **Run MoE on upload**.")
        st.stop()
    replay = st.session_state.uploaded_replay
    replay_key = f"upload|{fingerprint}"
if st.session_state.get("replay_key") != replay_key:
    st.session_state.replay_key = replay_key
    st.session_state.position = 0
    st.session_state.playing = False
    st.session_state.last_tick = time.monotonic()


@st.fragment(run_every="1s")
def replay_panel():
    if st.session_state.playing and time.monotonic() - st.session_state.last_tick >= 0.9:
        st.session_state.position = min(st.session_state.position + 1, len(replay) - 1)
        st.session_state.last_tick = time.monotonic()
        if st.session_state.position == len(replay) - 1:
            st.session_state.playing = False

    left, middle, right = st.columns([1, 1, 5])
    with left:
        if st.button("Pause" if st.session_state.playing else "Play", width="stretch"):
            st.session_state.playing = not st.session_state.playing
            st.session_state.last_tick = time.monotonic()
    with middle:
        if st.button("Next window", width="stretch", disabled=st.session_state.position >= len(replay)-1):
            st.session_state.position += 1
            st.session_state.last_tick = time.monotonic()
    with right:
        if len(replay) > 1:
            st.slider("Window", 0, len(replay)-1, key="position", format="%d")
        else:
            st.caption("One complete 32-second window")

    pos = st.session_state.position
    current = replay.iloc[pos]
    elapsed = float(current.start_sec + current.window_sec)
    st.markdown('<div class="context-strip">'
                f'<span class="record-name"><strong>{escape(dataset.upper())}</strong> / {escape(record_id)}</span>'
                f'<span>WINDOW <strong>{pos+1:02d} / {len(replay):02d}</strong></span>'
                f'<span><strong>{elapsed:.0f} S</strong> ELAPSED</span>'
                '</div>', unsafe_allow_html=True)

    neural_selected = source == "Saved recording" and model_choice.startswith("Neural")
    rr = current.neural_rr_bpm if neural_selected else current.moe_rr_bpm
    valid_rr = bool(np.isfinite(rr) and (not neural_selected or current.neural_observed_candidates > 0))
    if not valid_rr:
        rr = np.nan
    fusion = current.get("pred_all_quality_fusion", np.nan)
    primary, secondary = st.columns([1.25, 1], gap="large")
    with primary:
        display_rr = f"{rr:.1f}" if valid_rr else "—"
        st.markdown(
            f'<div class="reading-card{"" if valid_rr else " warning"}">'
            f'<div class="reading-top"><span>{"NEURAL MOE / EXPERIMENTAL" if neural_selected else "MASKED MOE / EXPLORATORY"}</span>'
            f'<span>{"CURRENT RR" if valid_rr else "NO ESTIMATE"}</span></div>'
            f'<div class="reading-value">{display_rr}<small>{"breaths / min" if valid_rr else ""}</small></div>'
            '<div class="reading-bottom">32-second window <span aria-hidden="true">↗</span></div></div>',
            unsafe_allow_html=True)
    with secondary:
        top_l, top_r = st.columns(2)
        with top_l:
            if neural_selected:
                fact_card("Masked MoE", f"{current.moe_rr_bpm:.1f} bpm" if np.isfinite(current.moe_rr_bpm) else "—")
            else:
                fact_card("Quality fusion", f"{fusion:.1f} bpm" if np.isfinite(fusion) else "—")
        with top_r:
            if neural_selected:
                fact_card("Aleatoric SD", f"{current.neural_sd_aleatoric_bpm:.2f} bpm")
            else:
                fact_card("Available experts", f"{int(current.available_experts)} / 14")
        low_l, low_r = st.columns(2)
        with low_l:
            if neural_selected:
                fact_card("Observed experts", f"{int(current.neural_observed_candidates)} / 14")
            else:
                quality = current.rqi_composite
                fact_card("Respiratory quality", f"{quality:.2f}" if np.isfinite(quality) else "—")
        with low_r:
            if neural_selected:
                fact_card("Imputed gate weight", f"{100*current.neural_imputed_gate_weight:.1f}%")
            elif show_reference and np.isfinite(current.reference_rr_bpm):
                fact_card("Entered RR" if source == "Upload CSV" else "Reference RR",
                          f"{current.reference_rr_bpm:.1f} bpm")
            else:
                fact_card("Uncertainty", "Uncalibrated")

    status_explanation = ("Experimental checkpoint · SD uncalibrated" if neural_selected else
                          "Exploratory inference · confidence unavailable")
    st.markdown(f'<div class="statusline{"" if valid_rr else " warn"}">'
                '<span class="statusdot" aria-hidden="true"></span><span>'
                f'<strong>{"Estimate available" if valid_rr else "Estimate unavailable"}</strong><span class="status-divider">/</span>'
                f'{status_explanation}'
                '</span></div>', unsafe_allow_html=True)

    if source == "Upload CSV":
        model = json.loads(model_text)
        breakdown = source_breakdown(current, model["candidate_columns"])
        leading_source = breakdown.loc[breakdown["Gate share (%)"].idxmax(), "Source"] if valid_rr else "None"
        st.markdown('<div class="source-heading"><div><div class="section-kicker">01 / SIGNAL SOURCES</div>'
                    '<h2>What shaped this estimate</h2></div>'
                    f'<span class="source-lead">PRIMARY INPUT <strong>{escape(leading_source.upper())}</strong></span>'
                    '</div>', unsafe_allow_html=True)
        source_columns = st.columns(3, gap="small")
        for column, item in zip(source_columns, breakdown.to_dict("records")):
            share = item["Gate share (%)"]
            rr_text = f'{item["Candidate RR (bpm)"]:.1f} bpm' if share > 0 else "—"
            spread = item["Spread (bpm)"]
            spread_text = f'{spread:.1f} bpm' if np.isfinite(spread) else "—"
            with column:
                st.markdown(
                    f'<div class="source-card{ " active" if share > 0 else ""}">'
                    f'<div class="source-card-top"><span>{escape(item["Source"])}</span>'
                    f'<span>{int(item["Candidates"])} estimates</span></div>'
                    f'<div class="source-share">{share:.0f}<small>% gate share</small></div>'
                    f'<div class="source-bar" aria-hidden="true"><span style="width:{share:.1f}%"></span></div>'
                    f'<div class="source-card-bottom"><span>Candidate RR <strong>{rr_text}</strong></span>'
                    f'<span>Spread <strong>{spread_text}</strong></span></div></div>',
                    unsafe_allow_html=True)
        st.caption("Candidate RR is before the model's shared correction. Spread is disagreement, "
                   "not calibrated uncertainty. A dash means fewer than two estimates; "
                   "combined estimates reuse ECG/PPG.")
        with st.container(border=True, key="wave_panel"):
            st.markdown('<div class="section-kicker">02 / SIGNALS</div>', unsafe_allow_html=True)
            st.subheader("Waveforms")
            wave_columns = st.columns(2)
            for column, (kind, recording) in zip(wave_columns, (("ECG", st.session_state.uploaded_ecg),
                                                                ("PPG", st.session_state.uploaded_ppg))):
                with column:
                    if recording is None:
                        st.caption(f"{kind} not uploaded")
                        continue
                    mask = (recording.time_s >= current.start_sec) & (recording.time_s < current.start_sec+32)
                    indexes = np.flatnonzero(mask)[::max(1, int(np.ceil(mask.sum()/1200)))]
                    if len(indexes):
                        wave = pd.DataFrame({kind: recording.values[indexes]}, index=recording.time_s[indexes])
                        st.line_chart(wave, height=210, x_label="Time (s)", y_label=f"{kind} amplitude",
                                      color="#0a7668" if kind == "ECG" else "#667e75")

    seen = replay.iloc[:pos+1].copy()
    seen["time_s"] = seen.start_sec + seen.window_sec
    if neural_selected:
        seen["selected_rr_bpm"] = seen.neural_rr_bpm.where(seen.neural_observed_candidates.gt(0))
        chart = seen.set_index("time_s")[["selected_rr_bpm", "moe_rr_bpm", "pred_all_quality_fusion"]].rename(
            columns={"selected_rr_bpm": "Neural checkpoint (experimental)", "moe_rr_bpm": "Masked MoE",
                     "pred_all_quality_fusion": "Quality fusion"})
        chart_colors = ["#0a7668", "#ac7650", "#667e75"]
    else:
        seen["selected_rr_bpm"] = seen.moe_rr_bpm
        chart = seen.set_index("time_s")[["moe_rr_bpm", "pred_all_quality_fusion"]].rename(
            columns={"moe_rr_bpm": "Masked MoE", "pred_all_quality_fusion": "Quality fusion"})
        chart_colors = ["#0a7668", "#ac7650"]
    if show_reference:
        chart["Reference RR"] = seen.set_index("time_s").reference_rr_bpm
        chart_colors.append("#425c78")
    with st.container(border=True, key="trend_panel"):
        st.markdown('<div class="section-kicker">01 / TREND</div>', unsafe_allow_html=True)
        st.subheader("Rate over time")
        st.line_chart(chart, height=300, x_label="Recording time (s)",
                      y_label="Respiratory rate (breaths/min)", color=chart_colors)

    lcol, rcol = st.columns([3, 2], gap="large")
    with lcol:
        with st.expander("Expert pathways"):
            model = json.loads(model_text)
            quality_by_candidate = dict(zip(model["candidate_columns"], model["quality_columns"]))
            if neural_selected:
                spec, _ = get_neural_package()
                candidates = spec["candidate_cols"]
                raw_values = np.asarray([current[name] for name in candidates], dtype=float)
                used_values = np.clip(np.where(np.isnan(raw_values), spec["candidate_medians"], raw_values), 3, 60)
                gate_weights = np.asarray(current.neural_gate_weights)
                st.caption("Missing candidates are median-imputed and can receive gate weight.")
            else:
                candidates = model["candidate_columns"]
                raw_values = np.asarray([current[name] for name in candidates], dtype=float)
                used_values = np.where((raw_values >= 3) & (raw_values <= 60), raw_values, np.nan)
                gate_weights = np.asarray(current.expert_weights)
                st.caption("Absent candidates receive zero gate weight.")
            expert_rows = pd.DataFrame({"Expert": candidates,
                                        "RR used (bpm)": used_values,
                                        "Source": ["Imputed" if np.isnan(value) else "Observed" for value in raw_values],
                                        "Quality": [current[quality_by_candidate[name]] for name in candidates],
                                        "Gate weight (%)": gate_weights * 100})
            expert_rows = expert_rows.sort_values("Gate weight (%)", ascending=False)
            st.dataframe(expert_rows, hide_index=True, width="stretch", height=310,
                         column_config={"RR used (bpm)": st.column_config.NumberColumn(format="%.1f"),
                                        "Quality": st.column_config.NumberColumn(format="%.2f"),
                                        "Gate weight (%)": st.column_config.NumberColumn(format="%.1f%%")})
    with rcol:
        with st.expander("Window details & limitations"):
            st.write(f"**Largest gate weight:** "
                     f"{current.neural_top_expert if neural_selected else (current.selected_expert or 'None')}")
            if valid_rr:
                st.write(f"**Weighted candidate RR:** "
                         f"{current.neural_weighted_base_bpm if neural_selected else current.weighted_base_bpm:.2f} bpm")
                st.write(f"**Model correction:** "
                         f"{current.neural_residual_bpm if neural_selected else current.residual_bpm:+.2f} bpm")
                if neural_selected:
                    st.write(f"**Aleatoric SD:** {current.neural_sd_aleatoric_bpm:.2f} bpm")
                    st.write(f"**Weight on imputed candidates:** {100*current.neural_imputed_gate_weight:.1f}%")
                    st.caption("This SD is an uncalibrated model output, not a confidence percentage "
                               "or validated prediction interval.")
                else:
                    st.write(f"**Expert disagreement:** {current.expert_disagreement_bpm:.2f} bpm")
                    st.caption("Disagreement is the gate-weighted spread of candidate RR values. "
                               "It is not a confidence percentage or a prediction interval; "
                               "a small value cannot establish accuracy.")
            st.write(f"**ECG available:** {'Yes' if current.has_ecg else 'No'}")
            st.write(f"**PPG available:** {'Yes' if current.has_ppg else 'No'}")
            st.write(f"**ECG quality:** {current.ecg_sqi_composite:.2f}" if np.isfinite(current.ecg_sqi_composite) else "**ECG quality:** unavailable")
            st.write(f"**PPG quality:** {current.ppg_sqi_composite:.2f}" if np.isfinite(current.ppg_sqi_composite) else "**PPG quality:** unavailable")
            if show_reference:
                st.write(f"**{'Entered real RR' if source == 'Upload CSV' else 'Reference RR'}:** {current.reference_rr_bpm:.1f} bpm")
                if valid_rr and np.isfinite(current.reference_rr_bpm):
                    st.write(f"**Absolute error:** {abs(rr-current.reference_rr_bpm):.1f} bpm")

    summary_frame = seen.copy()
    summary_frame["moe_rr_bpm"] = seen.selected_rr_bpm
    summary = summarize_replay(summary_frame)
    st.markdown('<div class="section-kicker" style="margin-top:1.8rem">02 / REPLAY</div>',
                unsafe_allow_html=True)
    summary_columns = st.columns(3 if show_reference else 2)
    x, y = summary_columns[:2]
    with x:
        mini_stat("Retained coverage", f"{summary['coverage']:.1%}")
    with y:
        mini_stat("Available / replayed", f"{summary['retained']} / {summary['windows']}")
    if show_reference:
        with summary_columns[2]:
            mini_stat("Error vs entered RR" if source == "Upload CSV" else "Descriptive MAE",
                      f"{summary['mae_bpm']:.2f} bpm" if summary['mae_bpm'] is not None else "—")

    export = seen[["record_id", "start_sec", "window_sec", "moe_rr_bpm", "status",
                   "available_experts", "selected_expert", "weighted_base_bpm", "residual_bpm",
                   "expert_disagreement_bpm", "pred_all_quality_fusion"]].copy()
    if neural_selected:
        export["neural_rr_bpm"] = seen.selected_rr_bpm
        export["neural_sd_aleatoric_bpm"] = seen.neural_sd_aleatoric_bpm
        export["neural_observed_candidates"] = seen.neural_observed_candidates
        export["neural_imputed_gate_weight"] = seen.neural_imputed_gate_weight
    if show_reference:
        export["reference_rr_bpm"] = seen.reference_rr_bpm
    st.download_button("Download replayed windows (CSV)", export.to_csv(index=False).encode(),
                       file_name=f"{dataset}_{record_id}_replay.csv", mime="text/csv")


replay_panel()
with st.expander("About this research interface"):
    st.caption("Playback advances by window, not real elapsed time. CapnoBase views are retrospective "
               "external audits and should not guide model settings.")
    st.caption("Neural checkpoint estimates use reconstructed preprocessing that has not been verified "
               "against the original run. Aleatoric SD is uncalibrated and is not a confidence score "
               "or validated prediction interval. Uploaded signal features are also exploratory.")
