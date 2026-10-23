# Respira — respiratory-rate research dashboard

Streamlit research app for estimating respiratory rate (RR) from timestamped ECG and/or PPG CSV recordings. It uses a compact physiology mixture-of-experts (MoE) with 14 candidate estimators. The dashboard shows ECG, PPG, and combined-source gate shares and descriptive candidate spread. The results are exploratory and are not validated for clinical decisions.

## Run locally

Use Python 3.12 or later, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_rr_dashboard/app.py
```

When the separate local study exports are present under `Data/rr_uncertainty_exports/rr_uncertainty_cpu_results`, the app also enables saved-recording replay and its experimental single neural MoE checkpoint. Those study exports are excluded from this public repository. Without them, the app starts directly in CSV upload mode.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
3. Create an app from the repository's `main` branch, with entrypoint `streamlit_rr_dashboard/app.py`. Choose Python 3.12 in Advanced settings.
4. Open the app URL and upload an ECG or PPG CSV with `time_s` and `ecg` or `ppg` columns.

No API keys or secrets are required. Uploaded recordings are processed on the Streamlit host; this code does not save them to the repository. Use de-identified or test recordings on a public deployment until its data-handling policy is reviewed. For file format, interpretation limits, and the local saved-recording mode, see the [dashboard README](streamlit_rr_dashboard/README.md).

The included neural package is a research artifact for the local saved-feature replay. Its original fitted preprocessor was not exported, so its predictions remain experimental. It is not applied to CSV uploads.
