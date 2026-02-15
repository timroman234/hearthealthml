# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Heart disease prediction ML project using logistic regression. Uses scikit-learn for ML and pandas/numpy for data processing.

## Commands

```bash
# Install dependencies
uv sync

# Run ML pipeline (standard)
uv run python main.py

# Run with hyperparameter tuning
uv run python main.py --tune

# Run with custom config
uv run python main.py --config configs/config.yaml

# Run tests
uv run pytest tests/
```

## Architecture

```
hearthealthml/
├── main.py                     # Pipeline entry point
├── configs/                    # YAML configuration files
├── data/
│   ├── raw/                    # Original immutable data
│   ├── processed/              # Cleaned/transformed data
│   └── splits/                 # Train/val/test splits
├── src/
│   ├── data/                   # loader, preprocessor, splitter
│   ├── features/               # build_features (engineering)
│   ├── models/                 # train, registry
│   ├── evaluation/             # evaluate (metrics, plots)
│   └── utils/                  # config, logger
├── models/                     # Saved model artifacts
├── reports/                    # Generated figures and metrics
├── notebooks/                  # Exploration notebooks
└── tests/                      # Unit tests
```

## Pipeline Stages (main.py)

1. **Data Loading** - Load CSV, validate schema
2. **Data Validation** - Missing values, outliers, range checks
3. **Feature Engineering** - age_group, bp_category, cardiac_risk_score, heart_rate_reserve, interactions
4. **Data Splitting** - Stratified 70/15/15 train/val/test
5. **Preprocessing** - StandardScaler (continuous), passthrough (binary), OneHotEncoder (categorical)
6. **Model Training** - Logistic regression (or with `--tune` for GridSearchCV)
7. **Threshold Optimization** - Find optimal classification threshold on validation set
8. **Evaluation** - Accuracy, precision, recall, F1, ROC-AUC, confusion matrix
9. **Model Registry** - Versioned artifacts with metadata

## Feature Categories

```python
CONTINUOUS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
BINARY = ['sex', 'fbs', 'exang']
CATEGORICAL = ['cp', 'restecg', 'slope', 'ca', 'thal']
TARGET = 'target'
```

## Configuration

Pipeline config in `configs/config.yaml`. See [ML_PIPELINE.md](ML_PIPELINE.md) for full design documentation.
