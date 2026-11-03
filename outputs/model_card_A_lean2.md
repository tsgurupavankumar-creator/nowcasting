# Cloudburst model card — Config A_lean2

**Config A_lean2 is the deployable model** (five features; comparison PR-AUC 0.2768, highest of the three atmospheric-only feature sets).

**Config A_lean (four features) is NOT the winner despite earlier guidance**; LogReg benefits from `wind_speed`, unlike XGBoost, which could reconstruct wind speed from `u10` and `v10` components. **Config A_full also underperforms A_lean2 by a small margin** (0.2747 vs. 0.2768).

The earlier F1=0.327 at threshold 0.82 was measured after tuning that threshold on the test set and is optimistic. It is not used as the deployable estimate here; the prior five-fold chronological training CV F1 at threshold 0.50 is 0.2520 ± 0.0507.

## Model

- **Estimator:** `LogisticRegression(class_weight='balanced', C=0.0022310108)`
- **Features:** t2m, u10, sp, tcwv, wind_speed
- **Split:** chronological 80% train / 20% test; training split further divided chronologically into 85% fit / 15% validation.
- **Fit rows:** 39,791; **validation rows:** 7,023; **test rows:** 11,704.
- **Fit period:** 2005-06-01 05:00:00 to 2018-08-12 03:00:00
- **Validation period:** 2018-08-12 04:00:00 to 2020-09-29 18:00:00
- **Test period:** 2020-09-29 19:00:00 to 2024-09-30 23:00:00

## Validation-selected thresholds

- **default:** 0.50 (pre-specified; not selected on validation).
- **t_F1:** 0.76 (validation precision=0.3271, recall=0.4023, F1=0.3608, balanced accuracy=0.6852)
- **t_recall_at_P70:** 0.92 (validation precision=0.7143, recall=0.0383, F1=0.0727, balanced accuracy=0.5189)
- **t_balanced:** 0.37 (validation precision=0.0997, recall=0.9157, F1=0.1798, balanced accuracy=0.7983)

## Test metrics (each threshold applied once)

Test PR-AUC from this fitted model: 0.2931.

| Threshold source | Threshold | Precision | Recall | F1 | BalAcc | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.50 | 0.1265 | 0.8384 | 0.2199 | 0.8013 | 8595/2651/74/384 |
| t_F1 | 0.76 | 0.2747 | 0.5153 | 0.3584 | 0.7299 | 10623/623/222/236 |
| t_recall_at_P70 | 0.92 | 0.4773 | 0.0917 | 0.1538 | 0.5438 | 11200/46/416/42 |
| t_balanced | 0.37 | 0.0957 | 0.9258 | 0.1734 | 0.7847 | 7238/4008/34/424 |

The model used to generate these metrics was fitted on the 85% fit portion only. The validation partition was used only for threshold selection; no test metric participated in choosing thresholds.

## Top SHAP features

1. tcwv: mean |SHAP| = 1.67341087
2. t2m: mean |SHAP| = 1.03837047
3. wind_speed: mean |SHAP| = 0.41549389
4. u10: mean |SHAP| = 0.20296774
5. sp: mean |SHAP| = 0.01879732

## Feature-set comparison

| Feature set | LogReg comparison PR-AUC |
|---|---:|
| A_full | 0.2747 |
| A_lean | 0.2553 |
| A_lean2 | 0.2768 |

## Rain-label leakage diagnostic

D4 found 38 of 47 equal-width `rain_3h` bins with a constant label and a transition band around 0.40–2.73. D3 found that `rain_peak_3h` alone achieved F1=0.931, precision=0.928, and recall=0.935. Rain-derived features are therefore excluded from this atmospheric-only deployable model.

The rain-derived reference configurations were not retrained for this corrected model.

## Operating recommendation

For nowcasting where missed events cost more than false alarms, use the validation F1-maximizing threshold. Consult the corresponding test row above for the observed recall, precision, and false-alarm burden. A validation-tuned operating point is still subject to temporal-regime drift and should be prospectively monitored.
- At the validation-selected F1 threshold, test precision=0.2747, or approximately 2.64 false-alarm hours per true hit.
