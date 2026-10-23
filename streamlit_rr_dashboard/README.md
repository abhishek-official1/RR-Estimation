# RR research replay dashboard

This local Streamlit app replays saved 32-second feature windows with a single checkpoint from the original neural physiology MoE or the compact masked MoE. It also accepts timestamped ECG/PPG CSV signals for exploratory inference with the masked MoE. It does not rerun the research notebook or train a model.

## Run

From the project root:

```powershell
python -m unittest discover -s streamlit_rr_dashboard -p 'test_*.py'
python -m streamlit run streamlit_rr_dashboard/app.py
```

Choose **Saved recording** or **Upload CSV** in the sidebar. Use Play, Next window, or the slider to navigate. The dashboard shows the MoE estimate, quality-fusion comparator, candidate estimates and gate weights, modality availability, quality, trend, and retained coverage. It also shows the gate-weighted candidate RR, bounded residual correction, and gate-weighted expert disagreement. Download the replayed rows as CSV.

The public GitHub/Streamlit deployment excludes the original study exports and starts in **Upload CSV** mode. Saved-recording replay appears only when the local feature tables and inventory are installed. The public upload path does not run the neural checkpoint.

The **Neural checkpoint (experimental)** choice loads `neural_package/weights.npz`, converted from the saved `moe_state.pt`, and applies a reconstructed median imputer and scaler. It runs on CPU with NumPy. To rebuild the package from existing processed feature tables, run `python -m streamlit_rr_dashboard.package_neural_moe` from the project root. Packaging takes seconds and performs no training.

## CSV upload

Upload separate CSV files for ECG and PPG. Either one can be omitted. A combined file can be selected in both upload fields if it has all three columns. Column names are case sensitive:

```csv
time_s,ecg
0.000,0.13
0.008,0.16
```

```csv
time_s,ppg
0.000,0.48
0.008,0.51
```

`time_s` is elapsed seconds on a shared clock. Each signal must contain at least 32 seconds of finite, nearly uniform samples at 20–1000 Hz. Each file is limited to 20 MB and each recording to 5 minutes. The app computes 32-second windows every 4 seconds where a complete signal window exists. It displays uploaded waveform segments alongside the MoE estimates.

To compare against a known value, check **I have a real/reference RR value** and enter one value in breaths/min (3–60). That value is treated as a constant reference for the entire uploaded recording. It is never used to produce MoE estimates; the dashboard reports descriptive error against it.

Each uploaded window shows ECG, PPG, and combined-source gate shares. The displayed candidate RR and within-source spread use the masked MoE's active expert weights before its shared residual correction. A combined candidate is attributed to the available modality when only one signal contributes. **Spread is descriptive disagreement, not calibrated uncertainty or a probability of RR accuracy.** A dash means fewer than two active candidates in that source.

## Scope and interpretation

- Saved-recording mode uses processed feature tables. Upload mode extracts approximate respiratory features from CSV waveforms. Live sensors are not available. Play advances one window per second rather than real elapsed time.
- Upload extraction uses SciPy peak detection and four classical RR estimators. The notebook used NeuroKit2 when available and also used an autoregressive estimator; those dependencies are absent here. Uploaded-signal estimates therefore need parity checks and subject-separated validation before scientific accuracy claims.
- The compact MoE has no calibrated uncertainty or selective threshold. The app marks a window **Unavailable** if it has no valid candidate; it does not claim a confidence or clinical safety decision.
- **Expert disagreement is descriptive, not confidence.** It is the gate-weighted standard deviation of the candidate RR estimates. Several candidates reuse the same physiological signals, so a small spread can still accompany a large RR error. Gate weights describe the model calculation, not causal proof that an expert is trustworthy. A confidence percentage or error interval would require subject-separated calibration and independent testing.
- The original notebook did not export its fitted imputer and scaler. Replaying its documented split on the available feature tables yields 7,990 training windows, while the saved study reports 8,237. The reconstructed preprocessor has the expected 195 context dimensions and 14 candidate experts, but its numeric statistics cannot be verified as the original values. **Neural checkpoint estimates are experimental and are not verified reproductions of the original neural run.** The saved ensemble member checkpoints are also unavailable, so the neural option is one checkpoint only.
- Uploads use `moe_improvement/results/prototype_model.json` and the masked MoE, because their feature extractor has not been checked for parity with the original neural model.
- Reference RR is only used for optional descriptive displays and never for prediction. CapnoBase is a **retrospective external audit** and must not guide parameter choices. A fresh cohort is needed for confirmation.
- To establish exact neural compatibility, obtain the original fitted preprocessor or reproduce the original training split and compare predictions on held-out windows. Then validate the upload feature extractor against saved features on matched windows and calibrate selective decisions on development subjects.
