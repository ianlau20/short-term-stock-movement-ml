# Short-Term Stock Movement Prediction (Machine Learning)

End-to-end machine learning project that predicts short-term stock price movements using historical market data. 
Implemented data preprocessing, feature engineering, model training (LSTM, XGBoost), and evaluation with ablation analysis.

**Tech stack:** Python, NumPy, Pandas, scikit-learn, XGBoost, PyTorch  
**Focus:** ML pipelines, model evaluation, reproducibility

## Project Highlights
- Built a reproducible ML pipeline for time-series classification using historical stock data
- Implemented and compared deep learning (LSTM) and tree-based (XGBoost) models
- Designed feature ablation experiments to analyze signal vs noise in financial indicators
- Evaluated models using F1 score, ROC-AUC, and confusion matrices
- Structured experiments and results for repeatability and analysis

## Engineering Takeaways
- Financial time-series prediction is a weak-signal problem where evaluation design matters more than model complexity
- Chronological data splits and ablation studies were critical to avoiding misleading results
- More expressive models (LSTM) improved class balance and ranking metrics even when accuracy gains were small

## Repository Structure
- `AML_final_code.ipynb` – Final model training and evaluation
- `aml_project_milestone.ipynb` – Intermediate experiments
- `conv_csv.py` – Metrics conversion and aggregation
- `*.csv` – Experiment results and ablation metrics
## Full Report
See [REPORT.md](./REPORT.md) for the complete academic writeup and experimental results.

