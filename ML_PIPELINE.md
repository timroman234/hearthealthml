# ML Pipeline Design - HeartHealthML

A modular, production-ready machine learning pipeline for heart disease prediction using logistic regression.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Directory Structure](#directory-structure)
4. [Pipeline Stages](#pipeline-stages)
5. [Feature Engineering](#feature-engineering)
6. [Configuration Schema](#configuration-schema)
7. [Pipeline Execution Flow](#pipeline-execution-flow)
8. [Model Registry](#model-registry)

---

## Overview

This pipeline implements a complete ML workflow for binary classification of heart disease, designed with:

- **Modularity**: Separated concerns across data, features, models, and evaluation
- **Extensibility**: Model registry pattern allows easy addition of new algorithms
- **Reproducibility**: Configuration-driven with versioned model artifacts
- **Production-readiness**: Proper logging, error handling, and validation

**Current Model**: Logistic Regression (baseline)
**Target Metric**: ROC-AUC (with emphasis on recall for medical context)

---

## Dataset Description

**Source**: UCI Heart Disease Dataset
**File**: `data/raw/heart_disease_data.csv`
**Samples**: 303
**Target**: Binary classification (0=healthy, 1=heart disease)

### Feature Catalog

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `age` | Continuous | 29-77 | Age in years |
| `sex` | Binary | 0, 1 | 0=female, 1=male |
| `cp` | Categorical | 0-3 | Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic) |
| `trestbps` | Continuous | 94-200 | Resting blood pressure (mm Hg) |
| `chol` | Continuous | 126-564 | Serum cholesterol (mg/dl) |
| `fbs` | Binary | 0, 1 | Fasting blood sugar > 120 mg/dl |
| `restecg` | Categorical | 0-2 | Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy) |
| `thalach` | Continuous | 71-202 | Maximum heart rate achieved |
| `exang` | Binary | 0, 1 | Exercise induced angina |
| `oldpeak` | Continuous | 0-6.2 | ST depression induced by exercise |
| `slope` | Categorical | 0-2 | Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping) |
| `ca` | Categorical | 0-4 | Number of major vessels colored by fluoroscopy |
| `thal` | Categorical | 0-3 | Thalassemia (0=normal, 1=fixed defect, 2=reversible defect, 3=unknown) |
| `target` | Binary | 0, 1 | Heart disease presence (target variable) |

### Feature Categories

```python
CONTINUOUS_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
BINARY_FEATURES = ['sex', 'fbs', 'exang']
CATEGORICAL_FEATURES = ['cp', 'restecg', 'slope', 'ca', 'thal']
TARGET = 'target'
```

---

## Directory Structure

```
hearthealthml/
├── data/
│   ├── raw/                    # Original immutable data
│   │   └── heart_disease_data.csv
│   ├── processed/              # Cleaned/transformed data
│   └── splits/                 # Train/val/test splits (CSV or pickle)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Data loading utilities
│   │   ├── preprocessor.py     # Feature preprocessing pipeline
│   │   └── splitter.py         # Train/val/test splitting
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py   # Feature engineering transformations
│   │   └── selection.py        # Feature selection methods
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # Model training logic
│   │   ├── predict.py          # Inference logic
│   │   └── registry.py         # Model registry/factory
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Custom metrics definitions
│   │   └── evaluate.py         # Model evaluation routines
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging utilities
│
├── configs/
│   ├── config.yaml             # Main configuration
│   ├── model_config.yaml       # Model hyperparameters
│   └── feature_config.yaml     # Feature definitions
│
├── models/                     # Saved model artifacts
│   └── .gitkeep
│
├── reports/
│   ├── figures/                # Generated plots (ROC, confusion matrix)
│   └── metrics/                # Evaluation results (JSON/CSV)
│
├── notebooks/
│   └── exploration.ipynb       # EDA notebook
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # Data loading/validation tests
│   ├── test_features.py        # Feature engineering tests
│   └── test_models.py          # Model training/prediction tests
│
├── scripts/
│   └── train_model.py          # Training script (CLI)
│
├── main.py                     # Full pipeline execution (entry point)
│
├── pyproject.toml
├── CLAUDE.md
├── ML_PIPELINE.md              # This document
└── README.md
```

---

## Pipeline Stages

### Stage 1: Data Loading

**Module**: `src/data/loader.py`

**Purpose**: Load raw data from CSV with validation.

**Functions**:
```python
def load_raw_data(path: Path) -> pd.DataFrame:
    """Load CSV data with error handling."""

def validate_schema(df: pd.DataFrame, expected_columns: list[str]) -> bool:
    """Validate DataFrame has expected columns and types."""

def get_data_info(df: pd.DataFrame) -> dict:
    """Return metadata: shape, dtypes, missing counts."""
```

**Validation Checks**:
- File exists and is readable
- Required columns present
- No unexpected columns
- Data types match expectations

---

### Stage 2: Data Validation & Quality Checks

**Module**: `src/data/preprocessor.py`

**Purpose**: Ensure data quality before processing.

**Functions**:
```python
def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return count of missing values per column."""

def detect_outliers(df: pd.DataFrame, columns: list[str], method: str = 'iqr') -> pd.DataFrame:
    """Detect outliers using IQR or Z-score method."""

def validate_ranges(df: pd.DataFrame, rules: dict) -> list[str]:
    """Validate values are within expected ranges."""
```

**Validation Rules**:
```python
VALIDATION_RULES = {
    'age': {'min': 0, 'max': 120},
    'sex': {'values': [0, 1]},
    'cp': {'values': [0, 1, 2, 3]},
    'trestbps': {'min': 0, 'max': 300},
    'chol': {'min': 0, 'max': 600},
    'fbs': {'values': [0, 1]},
    'restecg': {'values': [0, 1, 2]},
    'thalach': {'min': 0, 'max': 250},
    'exang': {'values': [0, 1]},
    'oldpeak': {'min': 0, 'max': 10},
    'slope': {'values': [0, 1, 2]},
    'ca': {'values': [0, 1, 2, 3, 4]},
    'thal': {'values': [0, 1, 2, 3]},
    'target': {'values': [0, 1]}
}
```

---

### Stage 3: Exploratory Data Analysis

**Module**: `notebooks/exploration.ipynb` (interactive) + `src/data/preprocessor.py` (automated)

**Analysis Tasks**:
1. **Distribution Analysis**: Histograms/KDE for continuous, bar charts for categorical
2. **Target Balance**: Check class imbalance ratio
3. **Correlation Analysis**: Feature-feature and feature-target correlations
4. **Statistical Tests**: Chi-square for categorical, t-test for continuous vs target

**Outputs**:
- `reports/figures/distributions.png`
- `reports/figures/correlation_heatmap.png`
- `reports/metrics/eda_summary.json`

---

### Stage 4: Data Preprocessing

**Module**: `src/data/preprocessor.py`

**Purpose**: Transform raw features into model-ready format.

**Transformations**:

| Feature Type | Transformer | Rationale |
|--------------|-------------|-----------|
| Continuous | `StandardScaler` | Zero mean, unit variance for logistic regression |
| Binary | `passthrough` | Already in 0/1 format |
| Categorical | `OneHotEncoder` | Create dummy variables |

**Implementation**:
```python
def create_preprocessor(config: dict) -> ColumnTransformer:
    """Create sklearn ColumnTransformer for full preprocessing."""

    continuous_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])

    return ColumnTransformer([
        ('continuous', continuous_transformer, CONTINUOUS_FEATURES),
        ('binary', 'passthrough', BINARY_FEATURES),
        ('categorical', categorical_transformer, CATEGORICAL_FEATURES)
    ])

def fit_transform(preprocessor: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    """Fit preprocessor and transform data."""

def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    """Save fitted preprocessor to disk."""
```

---

### Stage 5: Feature Engineering

**Module**: `src/features/build_features.py`

**Purpose**: Create derived features to improve model performance.

**Engineered Features**:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `age_group` | Bins: [0-40, 40-55, 55-70, 70+] | Age risk categories |
| `bp_category` | Based on trestbps thresholds | Clinical BP classification |
| `cholesterol_risk` | chol < 200: low, 200-239: borderline, 240+: high | Lipid risk levels |
| `heart_rate_reserve` | 220 - age - thalach | Exercise capacity metric |
| `st_risk_score` | oldpeak * (slope + 1) | Combined ST segment risk |
| `cardiac_risk_score` | Weighted sum of risk factors | Composite risk indicator |

**Implementation**:
```python
def create_age_groups(age: pd.Series) -> pd.Series:
    """Bin age into clinical categories."""
    bins = [0, 40, 55, 70, 120]
    labels = ['young', 'middle', 'senior', 'elderly']
    return pd.cut(age, bins=bins, labels=labels)

def create_bp_category(trestbps: pd.Series) -> pd.Series:
    """Classify blood pressure per clinical guidelines."""
    conditions = [
        trestbps < 120,
        (trestbps >= 120) & (trestbps < 130),
        (trestbps >= 130) & (trestbps < 140),
        trestbps >= 140
    ]
    categories = ['normal', 'elevated', 'high_stage1', 'high_stage2']
    return np.select(conditions, categories, default='unknown')

def create_cholesterol_risk(chol: pd.Series) -> pd.Series:
    """Classify cholesterol risk levels."""
    conditions = [chol < 200, (chol >= 200) & (chol < 240), chol >= 240]
    labels = ['desirable', 'borderline', 'high']
    return np.select(conditions, labels, default='unknown')

def calculate_heart_rate_reserve(age: pd.Series, thalach: pd.Series) -> pd.Series:
    """Calculate heart rate reserve (max predicted - achieved)."""
    max_predicted = 220 - age
    return max_predicted - thalach

def calculate_cardiac_risk_score(df: pd.DataFrame) -> pd.Series:
    """Calculate composite cardiac risk score."""
    score = (
        (df['age'] > 55).astype(int) * 1 +
        (df['sex'] == 1).astype(int) * 1 +
        (df['cp'] == 0).astype(int) * 2 +
        (df['trestbps'] >= 140).astype(int) * 1 +
        (df['chol'] >= 240).astype(int) * 1 +
        (df['fbs'] == 1).astype(int) * 1 +
        (df['exang'] == 1).astype(int) * 2 +
        (df['oldpeak'] > 2).astype(int) * 2 +
        (df['ca'] > 0).astype(int) * 2
    )
    return score

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    df = df.copy()
    df['age_group'] = create_age_groups(df['age'])
    df['bp_category'] = create_bp_category(df['trestbps'])
    df['cholesterol_risk'] = create_cholesterol_risk(df['chol'])
    df['heart_rate_reserve'] = calculate_heart_rate_reserve(df['age'], df['thalach'])
    df['cardiac_risk_score'] = calculate_cardiac_risk_score(df)
    return df
```

**Interaction Features**:
```python
def create_interaction_features(df: pd.DataFrame, pairs: list[tuple]) -> pd.DataFrame:
    """Create interaction terms between feature pairs."""
    df = df.copy()
    for f1, f2 in pairs:
        df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    return df

# Recommended interactions
INTERACTION_PAIRS = [
    ('age', 'thalach'),      # Age-heart rate interaction
    ('oldpeak', 'slope'),    # ST segment characteristics
    ('cp', 'exang'),         # Chest pain with exercise angina
]
```

---

### Stage 6: Feature Selection

**Module**: `src/features/selection.py`

**Purpose**: Identify most predictive features to reduce dimensionality.

**Methods**:

1. **Correlation Threshold**: Remove highly correlated features (>0.9)
2. **Variance Threshold**: Remove near-zero variance features
3. **Recursive Feature Elimination (RFE)**: Wrapper method using model
4. **L1 Regularization**: Embedded selection via Lasso coefficients

**Implementation**:
```python
def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    """Identify features to remove due to high correlation."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop

def select_by_variance(df: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    """Select features with variance above threshold."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    return df.columns[selector.get_support()].tolist()

def select_by_rfe(estimator, X: np.ndarray, y: np.ndarray, n_features: int) -> list[int]:
    """Select top N features using Recursive Feature Elimination."""
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    return np.where(rfe.support_)[0].tolist()

def select_by_importance(model, X: np.ndarray, y: np.ndarray, threshold: float = 0.01) -> list[int]:
    """Select features with importance above threshold."""
    model.fit(X, y)
    if hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    return np.where(importances > threshold)[0].tolist()
```

---

### Stage 7: Data Splitting

**Module**: `src/data/splitter.py`

**Purpose**: Create reproducible train/validation/test splits.

**Strategy**:
- **Train**: 70% - Model training
- **Validation**: 15% - Hyperparameter tuning
- **Test**: 15% - Final evaluation (held out)
- **Stratification**: Preserve target class distribution

**Implementation**:
```python
def create_splits(
    df: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> dict[str, pd.DataFrame]:
    """Create stratified train/val/test splits."""

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        stratify=y,
        random_state=random_state
    )

    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio_adjusted),
        stratify=y_temp,
        random_state=random_state
    )

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

def create_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42
) -> StratifiedKFold:
    """Create stratified K-Fold cross-validation iterator."""
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

def save_splits(splits: dict, output_dir: Path) -> None:
    """Save splits to disk as pickle files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        joblib.dump(data, output_dir / f'{name}.pkl')
```

---

### Stage 8: Model Training

**Module**: `src/models/train.py`

**Purpose**: Train models with configurable hyperparameters.

**Baseline Model**: Logistic Regression

**Implementation**:
```python
def get_model(model_name: str, config: dict) -> BaseEstimator:
    """Factory function to create model instances."""
    models = {
        'logistic_regression': LogisticRegression,
        # Future: 'random_forest': RandomForestClassifier,
        # Future: 'xgboost': XGBClassifier,
    }
    return models[model_name](**config)

def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
) -> BaseEstimator:
    """Train model and optionally evaluate on validation set."""
    model.fit(X_train, y_train)

    if X_val is not None:
        val_score = model.score(X_val, y_val)
        logger.info(f"Validation accuracy: {val_score:.4f}")

    return model

def save_model(model: BaseEstimator, metadata: dict, output_path: Path) -> None:
    """Save model artifact with metadata."""
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path / 'model.joblib')
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
```

---

### Stage 9: Hyperparameter Tuning

**Module**: `src/models/train.py`

**Purpose**: Optimize model hyperparameters via cross-validation.

**Logistic Regression Hyperparameter Space**:
```yaml
logistic_regression:
  C: [0.001, 0.01, 0.1, 1, 10, 100]
  penalty: ['l1', 'l2']
  solver: ['liblinear', 'saga']
  max_iter: [100, 200, 500]
  class_weight: [null, 'balanced']
```

**Implementation**:
```python
def tune_hyperparameters(
    model: BaseEstimator,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> dict:
    """Perform grid search with cross-validation."""
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_
    }
```

---

### Stage 10: Model Evaluation

**Module**: `src/evaluation/evaluate.py`

**Purpose**: Comprehensive model performance assessment.

**Metrics**:

| Metric | Purpose | Medical Context |
|--------|---------|-----------------|
| Accuracy | Overall correctness | General performance |
| Precision | Minimize false positives | Avoid unnecessary treatment |
| Recall | Minimize false negatives | **Critical**: Don't miss disease cases |
| F1-Score | Balanced P/R | Trade-off metric |
| ROC-AUC | Discrimination ability | Overall ranking quality |
| PR-AUC | Imbalanced performance | Handles class imbalance |
| Specificity | True negative rate | Correctly identify healthy |
| NPV | Negative predictive value | Confidence in negative predictions |

**Implementation**:
```python
def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """Compute all evaluation metrics."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path
) -> None:
    """Generate and save ROC curve plot."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def find_optimal_threshold(
    model: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimize_for: str = 'f1'
) -> float:
    """Find optimal classification threshold."""
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        if optimize_for == 'f1':
            scores.append(f1_score(y_val, y_pred))
        elif optimize_for == 'recall':
            scores.append(recall_score(y_val, y_pred))

    return thresholds[np.argmax(scores)]
```

**Medical Context Considerations**:
- Prioritize **recall** (sensitivity) to catch all disease cases
- Consider **cost-sensitive** learning where false negatives are more costly
- Generate **calibration plots** to assess probability reliability
- Report **confidence intervals** for metrics using bootstrapping

---

### Stage 11: Model Registry

**Module**: `src/models/registry.py`

**Purpose**: Version and track model artifacts.

**Storage Structure**:
```
models/
├── logistic_regression_v1.0.0/
│   ├── model.joblib           # Trained model
│   ├── preprocessor.joblib    # Fitted preprocessor
│   ├── metadata.json          # Training metadata
│   └── metrics.json           # Evaluation metrics
├── logistic_regression_v1.0.1/
│   └── ...
└── registry.json              # Model registry index
```

**Metadata Schema**:
```json
{
  "model_name": "logistic_regression",
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "hyperparameters": {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "max_iter": 200
  },
  "features_used": ["age", "sex", "cp", "..."],
  "feature_engineering": true,
  "training_samples": 212,
  "validation_samples": 45,
  "preprocessing_hash": "abc123"
}
```

**Implementation**:
```python
@dataclass
class ModelVersion:
    name: str
    version: str
    path: Path
    metrics: dict
    metadata: dict
    created_at: datetime

class ModelRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def register_model(
        self,
        model: BaseEstimator,
        preprocessor: ColumnTransformer,
        metrics: dict,
        metadata: dict
    ) -> str:
        """Register a new model version."""
        version = self._generate_version(metadata['model_name'])
        model_dir = self.registry_path / f"{metadata['model_name']}_v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save artifacts
        joblib.dump(model, model_dir / 'model.joblib')
        joblib.dump(preprocessor, model_dir / 'preprocessor.joblib')

        with open(model_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        metadata['version'] = version
        metadata['created_at'] = datetime.now().isoformat()
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        self._update_registry(metadata['model_name'], version, model_dir)
        return version

    def load_model(self, model_name: str, version: str = 'latest') -> tuple:
        """Load model and preprocessor by name and version."""
        if version == 'latest':
            version = self._get_latest_version(model_name)

        model_dir = self.registry_path / f"{model_name}_v{version}"
        model = joblib.load(model_dir / 'model.joblib')
        preprocessor = joblib.load(model_dir / 'preprocessor.joblib')

        return model, preprocessor

    def list_models(self) -> list[ModelVersion]:
        """List all registered models."""
        return list(self.registry.values())
```

---

## Configuration Schema

### Main Configuration (`configs/config.yaml`)

```yaml
# HeartHealthML Pipeline Configuration

data:
  raw_path: "data/raw/heart_disease_data.csv"
  processed_path: "data/processed/"
  splits_path: "data/splits/"

features:
  continuous:
    - age
    - trestbps
    - chol
    - thalach
    - oldpeak
  binary:
    - sex
    - fbs
    - exang
  categorical:
    - cp
    - restecg
    - slope
    - ca
    - thal
  target: target

  engineering:
    enabled: true
    create_age_groups: true
    create_bp_category: true
    create_cholesterol_risk: true
    create_heart_rate_reserve: true
    create_cardiac_risk_score: true
    interaction_features:
      - [age, thalach]
      - [oldpeak, slope]

preprocessing:
  scaler: standard          # standard, robust, minmax
  encoder: onehot           # onehot, ordinal
  handle_missing: drop      # drop, impute_mean, impute_median

splitting:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  random_state: 42
  stratify: true

training:
  default_model: logistic_regression
  cv_folds: 5
  random_state: 42

evaluation:
  primary_metric: roc_auc
  threshold: 0.5
  optimize_threshold: true
  threshold_metric: f1      # f1, recall, precision

output:
  models_dir: "models/"
  reports_dir: "reports/"
  figures_dir: "reports/figures/"
  metrics_dir: "reports/metrics/"

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pipeline.log"
```

### Model Configuration (`configs/model_config.yaml`)

```yaml
# Model Hyperparameter Configurations

logistic_regression:
  default:
    C: 1.0
    penalty: l2
    solver: liblinear
    max_iter: 200
    class_weight: null
    random_state: 42

  tuning:
    param_grid:
      C: [0.001, 0.01, 0.1, 1, 10, 100]
      penalty: [l1, l2]
      solver: [liblinear, saga]
      max_iter: [100, 200, 500]
      class_weight: [null, balanced]

    search_strategy: grid    # grid, random, bayesian
    cv_folds: 5
    scoring: roc_auc

# Future model configurations
# random_forest:
#   default:
#     n_estimators: 100
#     max_depth: 10
#     ...
```

### Feature Configuration (`configs/feature_config.yaml`)

```yaml
# Feature Definitions and Engineering Rules

feature_definitions:
  age:
    type: continuous
    description: "Age in years"
    range: [0, 120]

  sex:
    type: binary
    description: "Sex (0=female, 1=male)"
    values: [0, 1]

  cp:
    type: categorical
    description: "Chest pain type"
    values: [0, 1, 2, 3]
    labels:
      0: typical_angina
      1: atypical_angina
      2: non_anginal
      3: asymptomatic

  # ... (other features)

engineered_features:
  age_group:
    source: age
    type: categorical
    bins: [0, 40, 55, 70, 120]
    labels: [young, middle, senior, elderly]

  bp_category:
    source: trestbps
    type: categorical
    thresholds:
      normal: "<120"
      elevated: "120-129"
      high_stage1: "130-139"
      high_stage2: ">=140"

  cholesterol_risk:
    source: chol
    type: categorical
    thresholds:
      desirable: "<200"
      borderline: "200-239"
      high: ">=240"

  heart_rate_reserve:
    sources: [age, thalach]
    type: continuous
    formula: "(220 - age) - thalach"

  cardiac_risk_score:
    sources: [age, sex, cp, trestbps, chol, fbs, exang, oldpeak, ca]
    type: continuous
    description: "Composite cardiac risk score (0-13)"

feature_selection:
  correlation_threshold: 0.9
  variance_threshold: 0.01
  rfe_n_features: 10
```

---

## Pipeline Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Entry                           │
│                          main.py                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Data Loading                                          │
│  ├─ Load CSV from data/raw/                                     │
│  ├─ Validate schema                                             │
│  └─ Output: Raw DataFrame                                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Data Validation                                       │
│  ├─ Check missing values                                        │
│  ├─ Detect outliers                                             │
│  ├─ Validate value ranges                                       │
│  └─ Output: Validation report                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Feature Engineering                                   │
│  ├─ Create derived features (age_group, bp_category, etc.)     │
│  ├─ Create interaction features                                 │
│  └─ Output: Enhanced DataFrame                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Data Splitting                                        │
│  ├─ Stratified train/val/test split (70/15/15)                 │
│  ├─ Save splits to data/splits/                                 │
│  └─ Output: Split DataFrames                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: Preprocessing                                         │
│  ├─ Fit preprocessor on training data                          │
│  ├─ Transform all splits                                        │
│  ├─ Save fitted preprocessor                                    │
│  └─ Output: Transformed arrays                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 6: Feature Selection (Optional)                         │
│  ├─ Remove correlated features                                  │
│  ├─ Apply RFE or importance-based selection                    │
│  └─ Output: Selected feature indices                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 7: Model Training                                        │
│  ├─ Initialize model with config                               │
│  ├─ Train on training data                                      │
│  ├─ Validate on validation data                                 │
│  └─ Output: Trained model                                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 8: Hyperparameter Tuning                                 │
│  ├─ GridSearchCV with cross-validation                         │
│  ├─ Select best hyperparameters                                │
│  ├─ Retrain with best params                                    │
│  └─ Output: Tuned model                                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 9: Model Evaluation                                      │
│  ├─ Compute metrics on test set                                │
│  ├─ Generate confusion matrix                                   │
│  ├─ Generate ROC curve                                          │
│  ├─ Find optimal threshold                                      │
│  └─ Output: Metrics dict, plots                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 10: Model Registration                                   │
│  ├─ Save model artifact                                         │
│  ├─ Save preprocessor                                           │
│  ├─ Save metadata and metrics                                   │
│  ├─ Update registry                                             │
│  └─ Output: Model version ID                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Inference Pipeline

**Module**: `src/models/predict.py`

```python
def load_production_model(model_name: str = 'logistic_regression') -> tuple:
    """Load latest production model and preprocessor."""
    registry = ModelRegistry(Path('models/'))
    return registry.load_model(model_name, version='latest')

def predict_single(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    patient_data: dict
) -> dict:
    """Make prediction for a single patient."""
    df = pd.DataFrame([patient_data])
    X = preprocessor.transform(df)

    proba = model.predict_proba(X)[0, 1]
    prediction = int(proba >= 0.5)

    return {
        'prediction': prediction,
        'probability': float(proba),
        'risk_level': 'high' if proba >= 0.7 else 'medium' if proba >= 0.3 else 'low'
    }

def predict_batch(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame
) -> pd.DataFrame:
    """Make predictions for multiple patients."""
    X = preprocessor.transform(df)
    probas = model.predict_proba(X)[:, 1]
    predictions = (probas >= 0.5).astype(int)

    results = df.copy()
    results['prediction'] = predictions
    results['probability'] = probas
    return results
```

---

## Testing Strategy

**Module**: `tests/`

### Test Categories

1. **Data Tests** (`test_data.py`)
   - Schema validation
   - Missing value handling
   - Range validation

2. **Feature Tests** (`test_features.py`)
   - Feature engineering correctness
   - Transformation consistency
   - Edge cases

3. **Model Tests** (`test_models.py`)
   - Model training convergence
   - Prediction shape validation
   - Serialization/deserialization

### Example Tests

```python
# tests/test_data.py
def test_load_raw_data():
    df = load_raw_data(Path('data/raw/heart_disease_data.csv'))
    assert len(df) == 303
    assert 'target' in df.columns

def test_validate_schema():
    df = load_raw_data(Path('data/raw/heart_disease_data.csv'))
    assert validate_schema(df, EXPECTED_COLUMNS)

# tests/test_features.py
def test_age_groups():
    ages = pd.Series([25, 45, 60, 75])
    groups = create_age_groups(ages)
    assert list(groups) == ['young', 'middle', 'senior', 'elderly']

def test_cardiac_risk_score_range():
    df = load_raw_data(Path('data/raw/heart_disease_data.csv'))
    scores = calculate_cardiac_risk_score(df)
    assert scores.min() >= 0
    assert scores.max() <= 13

# tests/test_models.py
def test_model_training():
    model = get_model('logistic_regression', {'C': 1.0})
    X, y = make_classification(n_samples=100, random_state=42)
    trained = train_model(model, X, y)
    assert hasattr(trained, 'coef_')

def test_model_serialization():
    model = LogisticRegression().fit([[1], [2]], [0, 1])
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model(model, {}, Path(tmpdir))
        loaded = joblib.load(Path(tmpdir) / 'model.joblib')
        assert np.allclose(model.coef_, loaded.coef_)
```

---

## Dependencies

Required packages (add to `pyproject.toml`):

```toml
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "pyyaml>=6.0",
    "joblib>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

---

## Usage Examples

### Full Pipeline Run

```bash
# Install dependencies
uv sync

# Run complete pipeline
uv run python main.py

# Run with hyperparameter tuning
uv run python main.py --tune

# Run with custom config
uv run python main.py --config configs/config.yaml
```

### Training Only

```bash
uv run python scripts/train_model.py --model logistic_regression
```

### Making Predictions

```python
from src.models.predict import load_production_model, predict_single

model, preprocessor = load_production_model()

patient = {
    'age': 55,
    'sex': 1,
    'cp': 0,
    'trestbps': 140,
    'chol': 250,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.5,
    'slope': 1,
    'ca': 0,
    'thal': 2
}

result = predict_single(model, preprocessor, patient)
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

---

## Next Steps (Implementation Phase)

1. Create directory structure
2. Implement `src/data/loader.py`
3. Implement `src/data/preprocessor.py`
4. Implement `src/data/splitter.py`
5. Implement `src/features/build_features.py`
6. Implement `src/features/selection.py`
7. Implement `src/models/train.py`
8. Implement `src/models/predict.py`
9. Implement `src/models/registry.py`
10. Implement `src/evaluation/evaluate.py`
11. Implement `src/utils/config.py` and `logger.py`
12. Create configuration files
13. Implement `main.py`
14. Write unit tests
15. Move existing notebook to `notebooks/exploration.ipynb`
