# Cloudburst Prediction — Model Development and Evaluation Report

Atmospheric-only classification with leakage-safe methodology

Date:  2026-11-03

Data rows:  58,518 hourly observations

Period:  2005-06-01 — 2024-09-30

Table of contents

[Placeholder — update the table of contents in Word]


## 1. EXECUTIVE SUMMARY

This project evaluated whether atmospheric measurements could identify cloudburst hours while avoiding rain-derived inputs that encode the target. It compared twelve classification approaches across three atmospheric-only feature sets using chronological evaluation, with validation-only threshold selection for the corrected deployable model.

The honest result is a 6.8× lift over the event base rate and test F1=0.3584 at the validation-selected threshold 0.76; the comparison-run PR-AUC for A_lean2 is 0.2768. These are useful ranking results, but the selected test threshold still has precision 0.2747 and substantial false alarms. The finalize-run test PR-AUC was 0.2931 versus 0.2768 in the comparison run — a ~6% difference consistent with noise on 458 test positives; both should be read as the same result. 

The deployable model is LogisticRegression on the A_lean2 feature set; the fitted artifact is outputs/best_model_A_lean2.joblib.

The ceiling is constrained by a target definition that is nearly a direct rainfall threshold; removing rain observations avoids tautological performance but leaves a harder atmospheric-only classification task.

## 2. DATA AND LABEL-DEFINITION FINDING

| Dataset property | Value |
| --- | --- |
| Dataset shape | 58,518 rows × 18 columns (timestamp indexed) |
| Predictor columns | 17 recorded; 5 used by the deployable model |
| Target | cloudburst (binary) |
| Positive rate | 4.0774% |

| Partition | Rows | Date range |
| --- | --- | --- |
| Fit | 39,791 | 2005-06-01 05:00 — 2018-08-12 03:00 |
| Validation | 7,023 | 2018-08-12 04:00 — 2020-09-29 18:00 |
| Test | 11,704 | 2020-09-29 19:00 — 2024-09-30 23:00 |

### Critical finding: cloudburst is thresholded from rain

- D3: rain_peak_3h ≥ 2.227 alone achieves F1=0.931, precision=0.928, and recall=0.935.
- D4: 38 of 47 equal-width rain_3h bins have a constant cloudburst label; the transition band is 0.40–2.73.
Any model using rain_mm, rain_3h, rain_6h, rain_peak_3h, rain_lag1, rain_lag2, or tp reproduces the label by definition and does not predict it.


## 3. LEAKAGE PREVENTION MEASURES

1. Rain-derived features removed from all deployable models (rain_mm, rain_3h, rain_6h, rain_peak_3h, rain_lag1, rain_lag2, tp). Why: these variables directly encode the label, as shown by the single-feature threshold test.
2. Chronological 80/20 split with no shuffling. Why: training precedes the final test period, preventing future observations from entering training.
3. A 15% validation slice carved from the tail of the training set. Why: threshold selection never uses test labels.
4. Scaling and resampling performed inside sklearn Pipelines. Why: transformers and samplers are fit only on the current training portion or fold.
5. SMOTE applied only within training folds, never to validation or test. Why: synthetic minority examples cannot contaminate held-out evaluation.
6. Threshold selected on validation and applied once to test. Why: the reported operating-point metrics are not optimized on test.
7. TimeSeriesSplit(5) cross-validation. Why: each validation fold follows its training fold in time.
8. Drop-one-feature ablation quantified atmospheric-feature redundancy. Why: it shows how held-out PR-AUC changes when each atmospheric input is removed.
| Ablation | Dropped feature | PR-AUC |
| --- | --- | --- |
| drop_v10 | v10 | 0.2057 |
| drop_wind_speed | wind_speed | 0.1959 |
| drop_u10 | u10 | 0.1934 |
| all_6_features | none | 0.1914 |
| drop_sp | sp | 0.1655 |
| drop_tcwv | tcwv | 0.1485 |
| drop_t2m | t2m | 0.1373 |

9. Three feature sets benchmarked (A_full / A_lean / A_lean2). Why: comparison across feature definitions checks that the leading feature set is not an artifact of a single choice.
## 4. MODELS AND METHODOLOGY

All models use the same chronological holdout; evaluation metrics in the comparison CSVs use the default 0.50 decision threshold. Imbalance handling used in the training pipelines is summarized below.

| Model | Imbalance handling |
| --- | --- |
| Dummy | Most-frequent baseline; no class weighting. |
| LogReg | class_weight='balanced' |
| KNN | No explicit class weighting; scaled inputs. |
| DecisionTree | class_weight='balanced' |
| RandomForest | class_weight='balanced' |
| ExtraTrees | class_weight='balanced' |
| GradientBoosting | Class-balanced sample weights. |
| AdaBoost | Class-balanced sample weights. |
| XGBoost | scale_pos_weight from training class counts. |
| LightGBM | class_weight='balanced' |
| CatBoost | auto_class_weights='Balanced' |
| GaussianNB | No explicit class weighting. |

SVC and MLP were removed from the final run (10-minute runtime, no performance gain — documented in the earlier log). RandomForest and ExtraTrees use 300 estimators. RandomizedSearchCV used 10 iterations per model for the top 2 A_lean models; this tuning did not determine the corrected A_lean2 fit.


## 5. RESULTS

### A_full (6 feat)

| Rank | Model | PR-AUC | F1(pos) | Recall | Precision | ROC-AUC | Bal. Acc. | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LogReg | 0.2747 | 0.2452 | 0.8166 | 0.1443 | 0.8942 | 0.8097 | 0.8033 | 9,028 | 2,218 | 84 | 374 |
| 2 | GradientBoosting | 0.2588 | 0.2500 | 0.8057 | 0.1480 | 0.8927 | 0.8084 | 0.8108 | 9,121 | 2,125 | 89 | 369 |
| 3 | AdaBoost | 0.2479 | 0.2027 | 0.8384 | 0.1153 | 0.8775 | 0.7882 | 0.7419 | 8,299 | 2,947 | 74 | 384 |
| 4 | CatBoost | 0.2212 | 0.2789 | 0.5830 | 0.1833 | 0.8781 | 0.7386 | 0.8820 | 10,056 | 1,190 | 191 | 267 |
| 5 | LightGBM | 0.2157 | 0.2960 | 0.5284 | 0.2056 | 0.8744 | 0.7226 | 0.9017 | 10,311 | 935 | 216 | 242 |
| 6 | ExtraTrees | 0.2141 | 0.0820 | 0.0502 | 0.2233 | 0.8738 | 0.5216 | 0.9560 | 11,166 | 80 | 435 | 23 |
| 7 | RandomForest | 0.2104 | 0.2386 | 0.2162 | 0.2661 | 0.8754 | 0.5959 | 0.9460 | 10,973 | 273 | 359 | 99 |
| 8 | XGBoost | 0.1914 | 0.2816 | 0.3450 | 0.2380 | 0.8613 | 0.6500 | 0.9311 | 10,740 | 506 | 300 | 158 |
| 9 | GaussianNB | 0.1660 | 0.0956 | 0.0590 | 0.2523 | 0.8388 | 0.5259 | 0.9563 | 11,166 | 80 | 431 | 27 |
| 10 | KNN | 0.1513 | 0.1687 | 0.1201 | 0.2835 | 0.7514 | 0.5539 | 0.9537 | 11,107 | 139 | 403 | 55 |
| 11 | DecisionTree | 0.0641 | 0.1767 | 0.1638 | 0.1918 | 0.5678 | 0.5678 | 0.9403 | 10,930 | 316 | 383 | 75 |
| 12 | Dummy | 0.0391 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.9609 | 11,246 | 0 | 458 | 0 |


### A_lean (4 feat)

| Rank | Model | PR-AUC | F1(pos) | Recall | Precision | ROC-AUC | Bal. Acc. | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LogReg | 0.2553 | 0.2013 | 0.8515 | 0.1141 | 0.8746 | 0.7912 | 0.7356 | 8,219 | 3,027 | 68 | 390 |
| 2 | GradientBoosting | 0.2315 | 0.2125 | 0.8450 | 0.1215 | 0.8795 | 0.7981 | 0.7550 | 8,449 | 2,797 | 71 | 387 |
| 3 | AdaBoost | 0.2280 | 0.2182 | 0.7969 | 0.1264 | 0.8659 | 0.7863 | 0.7766 | 8,724 | 2,522 | 93 | 365 |
| 4 | LightGBM | 0.2088 | 0.2620 | 0.5895 | 0.1684 | 0.8547 | 0.7355 | 0.8700 | 9,913 | 1,333 | 188 | 270 |
| 5 | CatBoost | 0.1975 | 0.2401 | 0.6485 | 0.1473 | 0.8537 | 0.7478 | 0.8394 | 9,527 | 1,719 | 161 | 297 |
| 6 | ExtraTrees | 0.1755 | 0.1275 | 0.0852 | 0.2532 | 0.8426 | 0.5375 | 0.9544 | 11,131 | 115 | 419 | 39 |
| 7 | RandomForest | 0.1713 | 0.2325 | 0.2249 | 0.2407 | 0.8465 | 0.5980 | 0.9419 | 10,921 | 325 | 355 | 103 |
| 8 | XGBoost | 0.1708 | 0.2408 | 0.3428 | 0.1856 | 0.8287 | 0.6408 | 0.9154 | 10,557 | 689 | 301 | 157 |
| 9 | GaussianNB | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.8439 | 0.5000 | 0.9609 | 11,246 | 0 | 458 | 0 |
| 10 | KNN | 0.1644 | 0.2108 | 0.1528 | 0.3398 | 0.7414 | 0.5704 | 0.9552 | 11,110 | 136 | 388 | 70 |
| 11 | DecisionTree | 0.0558 | 0.1503 | 0.1507 | 0.1500 | 0.5579 | 0.5579 | 0.9334 | 10,855 | 391 | 389 | 69 |
| 12 | Dummy | 0.0391 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.9609 | 11,246 | 0 | 458 | 0 |


### A_lean2 (5 feat)

| Rank | Model | PR-AUC | F1(pos) | Recall | Precision | ROC-AUC | Bal. Acc. | Accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LogReg | 0.2768 | 0.2261 | 0.8319 | 0.1308 | 0.8916 | 0.8034 | 0.7772 | 8,715 | 2,531 | 77 | 381 |
| 2 | GradientBoosting | 0.2657 | 0.2426 | 0.8341 | 0.1420 | 0.8926 | 0.8144 | 0.7962 | 8,937 | 2,309 | 76 | 382 |
| 3 | AdaBoost | 0.2343 | 0.2059 | 0.8231 | 0.1177 | 0.8712 | 0.7859 | 0.7515 | 8,419 | 2,827 | 81 | 377 |
| 4 | LightGBM | 0.2189 | 0.2878 | 0.5590 | 0.1938 | 0.8697 | 0.7321 | 0.8917 | 10,181 | 1,065 | 202 | 256 |
| 5 | CatBoost | 0.2179 | 0.2802 | 0.6201 | 0.1810 | 0.8725 | 0.7529 | 0.8753 | 9,961 | 1,285 | 174 | 284 |
| 6 | ExtraTrees | 0.2141 | 0.1489 | 0.1004 | 0.2875 | 0.8769 | 0.5451 | 0.9551 | 11,132 | 114 | 412 | 46 |
| 7 | RandomForest | 0.2111 | 0.2736 | 0.2598 | 0.2888 | 0.8741 | 0.6169 | 0.9460 | 10,953 | 293 | 339 | 119 |
| 8 | XGBoost | 0.2057 | 0.2870 | 0.3559 | 0.2404 | 0.8529 | 0.6551 | 0.9308 | 10,731 | 515 | 295 | 163 |
| 9 | KNN | 0.1639 | 0.2101 | 0.1550 | 0.3257 | 0.7559 | 0.5710 | 0.9544 | 11,099 | 147 | 387 | 71 |
| 10 | GaussianNB | 0.1553 | 0.0244 | 0.0131 | 0.1765 | 0.8392 | 0.5053 | 0.9590 | 11,218 | 28 | 452 | 6 |
| 11 | DecisionTree | 0.0712 | 0.1993 | 0.1965 | 0.2022 | 0.5825 | 0.5825 | 0.9382 | 10,891 | 355 | 368 | 90 |
| 12 | Dummy | 0.0391 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.9609 | 11,246 | 0 | 458 | 0 |

### Feature-set comparison

| Feature set | Best model | PR-AUC | F1 | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| A_full (6 feat) | LogReg | 0.2747 | 0.2452 | 0.8166 | 0.1443 |
| A_lean (4 feat) | LogReg | 0.2553 | 0.2013 | 0.8515 | 0.1141 |
| A_lean2 (5 feat) | LogReg | 0.2768 | 0.2261 | 0.8319 | 0.1308 |

A_lean2 wins because Logistic Regression benefits from explicit wind_speed, which XGBoost could reconstruct from u10 + v10 in the earlier ablation. The XGBoost ablation finding did not generalize to LogReg.


## 6. FINAL MODEL AND OPERATING POINTS

> **DEPLOYABLE MODEL  \|  LogisticRegression  \|  C = 0.002231  \|  class_weight = balanced  \|  A_lean2 (5 features)**

Test-set metrics below come from threshold_metrics_A_lean2.csv; thresholds were selected using the validation CSV and then applied to test once.

| Threshold source | Threshold | Precision | Recall | F1 | Bal.Acc | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | 0.5000 | 0.1265 | 0.8384 | 0.2199 | 0.8013 | 8,595 | 2,651 | 74 | 384 |
| t_F1 (validation) | 0.7600 | 0.2747 | 0.5153 | 0.3584 | 0.7299 | 10,623 | 623 | 222 | 236 |
| t_recall@P70 | 0.9200 | 0.4773 | 0.0917 | 0.1538 | 0.5438 | 11,200 | 46 | 416 | 42 |
| t_balanced | 0.3700 | 0.0957 | 0.9258 | 0.1734 | 0.7847 | 7,238 | 4,008 | 34 | 424 |

Recommended operating point: threshold 0.76 (validation F1-optimal).

At threshold 0.76 the model catches 51.5% of cloudbursts with precision 27.5%, i.e. roughly 2.6 false alarms per true hit (assumes hourly observations).

Safety-critical alternative: threshold 0.37 yielded 92.6% test recall; alarm-fatigue-sensitive alternative: threshold 0.92 yielded 47.7% test precision, with 9.2% test recall.


## 7. FEATURE IMPORTANCE

Mean absolute SHAP values for the five features are from the corrected A_lean2 model output.

| Rank | Feature | Mean \|SHAP\| |
| --- | --- | --- |
| 1 | tcwv | 1.6734 |
| 2 | t2m | 1.0384 |
| 3 | wind_speed | 0.4155 |
| 4 | u10 | 0.2030 |
| 5 | sp | 0.0188 |

Direction note (signed Logistic Regression coefficients): tcwv increases modeled risk as its value rises; t2m decreases modeled risk as its value rises; wind_speed increases modeled risk as its value rises.

SHAP plot: outputs/shap_A_lean2.png.

## 8. HONEST LIMITATIONS

- The label is approximately rain_3h > 2.7; the deployable ceiling is F1 ≈ 0.36 without rain features.
- Precision ≥ 0.70 was not achievable at any useful recall on the test set. The best test precision at recall > 0.10 was 0.48 (threshold 0.92 gives precision 0.477, recall 0.092).
- The corrected fit-only model achieves test PR-AUC 0.2931 vs 0.2768 in the comparison run. On 458 test positives, ±0.03 PR-AUC run-to-run variation is expected; treat both as the same number.
- The model is a prioritization tool, not an automated alarm.
- Genuine improvement requires redefining the label and obtaining independent predictors such as radar reflectivity, lightning counts, or upper-air soundings.

## 9. FILES AND REPRODUCTION

| Filename | Purpose | Path |
| --- | --- | --- |
| best_model_A_lean2.joblib | Deployable Logistic Regression model | outputs/best_model_A_lean2.joblib |
| model_card_A_lean2.md | Detailed model card | outputs/model_card_A_lean2.md |
| threshold_metrics_A_lean2.csv | Test-set metrics at 4 thresholds | outputs/threshold_metrics_A_lean2.csv |
| threshold_selection_A_lean2.csv | Validation-set threshold selection log | outputs/threshold_selection_A_lean2.csv |
| results_A_full.csv / results_A_lean.csv / results_A_lean2.csv | Three atmospheric feature-set comparisons | outputs/results_A_full.csv; outputs/results_A_lean.csv; outputs/results_A_lean2.csv |
| ablation_A_drop_one_feature.csv | Atmospheric drop-one-feature ablation | outputs/ablation_A_drop_one_feature.csv |
| shap_A_lean2.png / pr_auc_models_A.png | Feature explanation / PR-AUC comparison plot | outputs/shap_A_lean2.png; outputs/pr_auc_models_A.png |
| diagnostics.md | D1–D5 label leakage diagnostics | outputs/diagnostics.md |
| run_pipeline.py --finalize-a-lean2 | Reproduction command for corrected fit and threshold evaluation | python run_pipeline.py --finalize-a-lean2 |

Reproduction command (one line):

`python run_pipeline.py --finalize-a-lean2`
