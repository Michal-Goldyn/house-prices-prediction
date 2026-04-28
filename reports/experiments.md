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


## Feature Engineering (03_feature_engineering.ipynb)
Tested 6 engineered features, then iterated to minimal subset based on RF feature importance.

### Initial: 6 features (TotalSF, TotalBath, RemodAge, HasPool, HasGarage, HasBsmt)

| Model             | RMSE   | Diff vs baseline |
|---|---|---|
| Decision Tree     | 0.2117 | +0.0044 (worse)  |
| Random Forest     | 0.1367 | -0.0030 (better) |
| Ridge             | 0.1289 | -0.0010 (better) |

### Feature importance from RF showed:
- TotalSF: 0.309 (top 2, dominant)
- RemodAge, TotalBath: ~0.01 (minor)
- HasPool, HasGarage, HasBsmt: not in top 20 (noise)

### Final: 1 feature (TotalSF only)

| Model         | RMSE          | Diff vs baseline |
|---|---|---|
| Decision Tree | 0.2088        | +0.0015          |
| Random Forest | 0.1372        | -0.0025 (better) |
| Ridge         | **0.1289**    | -0.0010 (better) |

TotalSF (sum of 1stFlrSF + 2ndFlrSF + TotalBsmtSF) does all the work. 
Other features added noise without any gains.
**New floor: 0.1289 RMSE (Ridge with TotalSF).**