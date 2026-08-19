<!-- simplex-warmstart-report -->
## Retrain Report — ✅ Admissible model

### Quality gate
| Control | Value | Threshold | Verdict |
| --- | --- | --- | --- |
| test_rmse | 0.1685 | 0.19 | 🟩
| test_r2 | 0.9602 | 0.9 | 🟩
| family_gap (esters) | 0.0086 | 0.1 | 🟩

### Metrics diff (vs `master`)
| Path               | Metric              | master   | workspace   | Change   |
|--------------------|---------------------|----------|-------------|----------|
| metrics\train.json | best_val_rmse       | 0.17772  | 0.17534     | -0.00238 |
| metrics\eval.json  | test.mae            | 0.13236  | 0.13119     | -0.00117 |
| metrics\eval.json  | test.r2             | 0.95985  | 0.96019     | 0.00034  |
| metrics\eval.json  | test.rmse           | 0.16925  | 0.16854     | -0.00071 |
| metrics\eval.json  | test_esters.mae     | 0.13746  | 0.13693     | -0.00053 |
| metrics\eval.json  | test_esters.r2      | 0.92503  | 0.92403     | -0.00099 |
| metrics\eval.json  | test_esters.rmse    | 0.17601  | 0.17717     | 0.00116  |
| metrics\eval.json  | test_silicones.mae  | 0.12828  | 0.1266      | -0.00169 |
| metrics\eval.json  | test_silicones.r2   | 0.92792  | 0.92996     | 0.00204  |
| metrics\eval.json  | test_silicones.rmse | 0.16365  | 0.16131     | -0.00233 |

### Performance per family
| Family | RMSE | MAE | R² |
| --- | --- | --- | --- |
| esters | 0.1772 | 0.1369 | 0.9240 |
| silicones | 0.1613 | 0.1266 | 0.9300 |

<details><summary>Data manifest</summary>

| Batch | Studies | Families | Seed |
| --- | --- | --- | --- |
| 0 | 120 | esters silicones | 0 |

</details>
