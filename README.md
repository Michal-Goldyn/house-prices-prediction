# House Prices: Advanced Regression Techniques

ML pipeline predicting Ames (Iowa, USA) housing prices using LightGBM + Optuna tuning.
Top 27.5% on Kaggle public leaderboard.

🏆 **Kaggle RMSLE: 0.12753** | **Position: 1409 / 5119 teams**

![Kaggle submission](reports/figures/06_kaggle_score.png)

## Pipeline

```mermaid
flowchart LR
    A[train.csv<br/>1460 x 81] --> B[Drop outliers<br/>GrLivArea>4000 & Price<300k]
    B --> C[log1p target<br/>SalePrice]
    C --> D[Feature Engineering<br/>+ TotalSF]
    D --> E[Fillna + OneHotEncoding]
    E --> F{5 models<br/>5-fold CV}
    F --> G[Decision Tree]
    F --> H[Random Forest]
    F --> I[XGBoost]
    F --> J[Ridge]
    F --> K[LightGBM 🏆]
    K --> L[Optuna tuning<br/>50 trials]
    L --> M[SHAP analysis]
    L --> N[Kaggle submission<br/>RMSLE 0.12753]
```