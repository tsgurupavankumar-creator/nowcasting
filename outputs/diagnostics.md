# Cloudburst leakage diagnostics (D1-D5)

- Dataset: `cleaned_cloudburst (2).csv`
- Rows: 58,518; positives: 2,386 (4.0774%)
- Diagnostics only; no model training was run.

## D1. Pearson correlation matrix

```text
              rain_mm  rain_3h  rain_6h  rain_peak_3h  rain_lag1  rain_lag2       tp  cloudburst
rain_mm      1.000000 0.969539 0.894417      0.963629   0.956984   0.886377 0.540093    0.724707
rain_3h      0.969539 1.000000 0.960938      0.990136   0.993615   0.969539 0.553065    0.741727
rain_6h      0.894417 0.960938 1.000000      0.954182   0.948249   0.975471 0.561626    0.725705
rain_peak_3h 0.963629 0.990136 0.954182      1.000000   0.979603   0.960533 0.545391    0.738704
rain_lag1    0.956984 0.993615 0.948249      0.979603   1.000000   0.956984 0.543406    0.732145
rain_lag2    0.886377 0.969539 0.975471      0.960533   0.956984   1.000000 0.538470    0.718405
tp           0.540093 0.553065 0.561626      0.545391   0.543406   0.538470 1.000000    0.411391
cloudburst   0.724707 0.741727 0.725705      0.738704   0.732145   0.718405 0.411391    1.000000
```

Pairs with |corr| > 0.95:
- rain_mm -- rain_3h: 0.969539
- rain_mm -- rain_peak_3h: 0.963629
- rain_mm -- rain_lag1: 0.956984
- rain_3h -- rain_6h: 0.960938
- rain_3h -- rain_peak_3h: 0.990136
- rain_3h -- rain_lag1: 0.993615
- rain_3h -- rain_lag2: 0.969539
- rain_6h -- rain_peak_3h: 0.954182
- rain_6h -- rain_lag2: 0.975471
- rain_peak_3h -- rain_lag1: 0.979603
- rain_peak_3h -- rain_lag2: 0.960533
- rain_lag1 -- rain_lag2: 0.956984

## D2. Rain feature ranges by target class

| Feature | cloudburst=0 min / median / max | cloudburst=1 min / median / max |
|---|---:|---:|
| rain_mm | -0.618649 / -0.411624 / 2.669816 | -0.363998 / 3.194179 / 14.195471 |
| rain_3h | -0.632848 / -0.411709 / 2.667911 | 0.515031 / 3.223584 / 12.313668 |

Range overlap assessment:
- rain_mm: class ranges overlap.
- rain_3h: class ranges overlap.

## D3. Best single-feature threshold rules

Rules tested: feature >= threshold and feature <= threshold, using each observed unique feature value as a cutoff.

| Feature | Best direction | Threshold | F1(pos) | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| rain_mm | >= | 2.61936679 | 0.850618 | 0.983498 | 0.749371 |
| rain_3h | >= | 2.33938296 | 0.879069 | 0.905016 | 0.854568 |
| rain_6h | >= | 2.62592042 | 0.851887 | 0.990556 | 0.747276 |
| rain_peak_3h | >= | 2.22729799 | 0.931301 | 0.928007 | 0.934619 |
| rain_lag1 | >= | 2.31663029 | 0.844072 | 0.865639 | 0.823554 |
| rain_lag2 | >= | 2.41703327 | 0.816281 | 0.869668 | 0.769070 |
| tp | >= | 2.22701319 | 0.380442 | 0.407071 | 0.357083 |

Tautology threshold (F1 >= 0.9):
- TAUTOLOGY CONFIRMED: rain_peak_3h alone reproduces cloudburst (rule `rain_peak_3h >= 2.22729799`, F1=0.931301; precision=0.928007, recall=0.934619).

## D4. Rain-triple uniqueness and rain_3h bins

- Unique `(rain_mm, rain_3h, cloudburst)` triples: 58,511 of 58,518 rows.
- Bin summary: 47 non-empty equal-width bins across the observed `rain_3h` range (nominal width 0.25893); 38/47 bins have a constant label.

| rain_3h bin | Rows | Positives | Positive rate | Label constant? |
|---|---:|---:|---:|---|
| (-0.634, -0.374] | 30,079 | 0 | 0.0000% | yes |
| (-0.374, -0.115] | 8,951 | 0 | 0.0000% | yes |
| (-0.115, 0.144] | 5,248 | 0 | 0.0000% | yes |
| (0.144, 0.403] | 3,340 | 0 | 0.0000% | yes |
| (0.403, 0.662] | 2,469 | 3 | 0.1215% | no |
| (0.662, 0.921] | 1,709 | 2 | 0.1170% | no |
| (0.921, 1.18] | 1,278 | 6 | 0.4695% | no |
| (1.18, 1.439] | 1,089 | 21 | 1.9284% | no |
| (1.439, 1.698] | 796 | 49 | 6.1558% | no |
| (1.698, 1.956] | 612 | 66 | 10.7843% | no |
| (1.956, 2.215] | 495 | 126 | 25.4545% | no |
| (2.215, 2.474] | 425 | 189 | 44.4706% | no |
| (2.474, 2.733] | 348 | 245 | 70.4023% | no |
| (2.733, 2.992] | 263 | 263 | 100.0000% | yes |
| (2.992, 3.251] | 251 | 251 | 100.0000% | yes |
| (3.251, 3.51] | 201 | 201 | 100.0000% | yes |
| (3.51, 3.769] | 172 | 172 | 100.0000% | yes |
| (3.769, 4.028] | 134 | 134 | 100.0000% | yes |
| (4.028, 4.287] | 108 | 108 | 100.0000% | yes |
| (4.287, 4.546] | 88 | 88 | 100.0000% | yes |
| (4.546, 4.805] | 72 | 72 | 100.0000% | yes |
| (4.805, 5.064] | 68 | 68 | 100.0000% | yes |
| (5.064, 5.323] | 53 | 53 | 100.0000% | yes |
| (5.323, 5.581] | 26 | 26 | 100.0000% | yes |
| (5.581, 5.84] | 44 | 44 | 100.0000% | yes |
| (5.84, 6.099] | 30 | 30 | 100.0000% | yes |
| (6.099, 6.358] | 25 | 25 | 100.0000% | yes |
| (6.358, 6.617] | 20 | 20 | 100.0000% | yes |
| (6.617, 6.876] | 20 | 20 | 100.0000% | yes |
| (6.876, 7.135] | 18 | 18 | 100.0000% | yes |
| (7.135, 7.394] | 13 | 13 | 100.0000% | yes |
| (7.394, 7.653] | 14 | 14 | 100.0000% | yes |
| (7.653, 7.912] | 7 | 7 | 100.0000% | yes |
| (7.912, 8.171] | 7 | 7 | 100.0000% | yes |
| (8.171, 8.43] | 6 | 6 | 100.0000% | yes |
| (8.43, 8.689] | 5 | 5 | 100.0000% | yes |
| (8.689, 8.948] | 3 | 3 | 100.0000% | yes |
| (8.948, 9.207] | 6 | 6 | 100.0000% | yes |
| (9.207, 9.465] | 5 | 5 | 100.0000% | yes |
| (9.465, 9.724] | 4 | 4 | 100.0000% | yes |
| (9.724, 9.983] | 2 | 2 | 100.0000% | yes |
| (9.983, 10.242] | 5 | 5 | 100.0000% | yes |
| (10.242, 10.501] | 2 | 2 | 100.0000% | yes |
| (10.76, 11.019] | 1 | 1 | 100.0000% | yes |
| (11.019, 11.278] | 3 | 3 | 100.0000% | yes |
| (11.278, 11.537] | 1 | 1 | 100.0000% | yes |
| (12.055, 12.314] | 2 | 2 | 100.0000% | yes |

## D5. Candidate atmospheric / temporal correlation ranking

| Rank | Feature | Pearson correlation with cloudburst |
|---:|---|---:|
| 1 | tcwv_3h | 0.195546 |
| 2 | tcwv | 0.195173 |
| 3 | tcwv_6h | 0.194741 |
| 4 | u10 | -0.150258 |
| 5 | t2m | -0.086489 |
| 6 | v10 | 0.047202 |
| 7 | sp | -0.046960 |
| 8 | sp_drop_3h | 0.017422 |
| 9 | wind_speed | 0.001288 |
| 10 | t2m_grad | -0.000263 |

Ranking is by absolute correlation; signs are preserved in the reported values.
