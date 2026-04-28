# Experiments log

## Baseline (notebook 02_baseline.ipynb)

Brutal preprocessing (median fill, 'Missing' fill, one-hot), no feature engineering, default hyperparameters.
5-fold CV, RMSE on log1p(SalePrice).

| Model         | RMSE      | Std    |
|-------        |------     |------- |
| Decision Tree | 0.2073    | 0.0096 |
| Random Forest | 0.1397    | 0.0101 |
| Ridge         | 0.1299    | 0.0133 |

**Floor for next experiments: 0.1299 (Ridge).**

### Observations
- DT overfits heavily - single tree memorizes training data.
- RF improves a lot.
- Ridge wins - likely because:
  - Linear relationships dominate (OverallQual, GrLivArea correlations)
  - High-dimensional after OHE (~260 features)
  - Tree-based may need feature engineering or tuning to compete.

### Next steps
- Add proper preprocessing (3-group missing strategy from EDA)
- Feature engineering (TotalSF, HouseAge, log skewed features)
- Test XGBoost + LightGBM
- Optuna tuning