# Cloudburst benchmark — Config A

Only scientifically valid atmospheric feature sets are trained:

- `A_full`: `t2m`, `u10`, `v10`, `sp`, `tcwv`, `wind_speed`
- `A_lean`: `t2m`, `u10`, `sp`, `tcwv`
- `A_lean2`: `t2m`, `u10`, `sp`, `tcwv`, `wind_speed`

Leaky reference configurations B–D are deliberately skipped. SVC and MLP are
also omitted for runtime and tabular-model considerations.

Run the three Config A feature sets with:

```powershell
python run_pipeline.py --config A
```

The run uses a chronological 80/20 split, five-fold TimeSeriesSplit CV, and
top-two-only SMOTE/tuning on A_lean. Outputs include `results_A_full.csv`,
`results_A_lean.csv`, `results_A_lean2.csv`, `best_model_A.joblib`,
`model_card_A.md`, and `shap_A.png`.

