# Solar Radiation Prediction using Saudi Arabia Dataset

[![CI/CD](https://github.com/Aamir-Sarang31/Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/Aamir-Sarang31/Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomizedSearchCV-F7931E.svg)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20%26%20Model%20Versioning-945DD6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Paper & Project Context:** Solar Irradiance Forecasting & Renewable Energy Predictive Modeling using Multi-Station Meteorological Observations across Saudi Arabia.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Benchmark Results](#-controlled-model-performance--benchmarks-10-fold-cross-validation)
- [Deep Learning Architectures](#-deep-learning-architectures)
- [Hyperparameter Tuning (RandomizedSearchCV)](#-automated-hyperparameter-tuning-randomizedsearchcv)
- [Data & Model Versioning (DVC)](#-data--model-versioning-dvc)
- [MLOps Pipeline](#-mlops-mlflow-experiment-tracking--registry-gating)
- [Project Structure](#-project-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Web Application](#-web-application)
- [Dataset](#-dataset)
- [CI/CD Pipeline](#-cicd-workflows)
- [Testing](#-testing)
- [Authors & Acknowledgements](#-authors--acknowledgements)
- [License](#-license)

---

## 🌟 Project Overview

This project forecasts **Global Horizontal Irradiance (GHI)** using meteorological observations collected from **43 weather monitoring stations across Saudi Arabia (2017–2021)**.

We systematically evaluate and benchmark **11 machine learning and deep learning models** under rigorous **10-Fold Cross-Validation** and **43-Fold Leave-One-Station-Out (LOSO) Cross-Validation**, with:

- 🧠 **3 Custom Deep Learning Architectures** (FT-Transformer, LSTM, 1D CNN)
- 📊 **8 Classical ML Baselines** (SVR, HGB, XGBoost, RF, ANN, LR, DT, KNN)
- 🔬 **Automated Hyperparameter Optimization** via Scikit-Learn `RandomizedSearchCV`
- 📦 **Data & Model Version Control** via DVC (Data Version Control)
- 📈 **End-to-end MLflow Experiment Tracking** with automated Model Registry promotion gating
- 🌐 **Interactive Flask Web Application** with real-time predictions and station maps
- ⚙️ **CI/CD Pipelines** with GitHub Actions

---

## 🏆 Key Features

| Feature | Description |
|---|---|
| **11-Model Controlled Benchmark** | All models evaluated under identical 10-fold CV partitions, same seed, and uniform scaling |
| **FT-Transformer Architecture** | Feature Tokenizer Transformer achieving **R² = 0.9919** — best-in-class for tabular solar data |
| **RandomizedSearchCV Tuning** | Scikit-Learn randomized search across continuous distributions for classical & PyTorch models |
| **Data & Model Versioning (DVC)** | DVC tracking for `dataset.csv` and model weights with reproducible `dvc.yaml` pipelines |
| **MLflow Experiment Tracking** | Every run logs parameters, metrics, plots, and model artifacts to SQLite |
| **Model Registry Promotion Gate** | Automated quality gate blocks deployment if candidate fails RMSE/R² thresholds |
| **43-Station LOSO Validation** | Spatial generalization testing across all Saudi Arabian weather stations |
| **Interactive Web App** | Flask app with Leaflet.js station map, multi-model predictions, and Plotly visualizations |
| **CI/CD Integration** | GitHub Actions for linting, testing, smoke training, and promotion gate verification |

---

## 🧪 Controlled Model Performance & Benchmarks (10-Fold Cross-Validation)

All 11 models evaluated under the **exact same 10-fold CV partitions, same random seed (42), and uniform feature/target scaling** on `dataset.csv` (1,649 records across 43 stations). Sorted strictly by ascending RMSE (primary gate metric):

| Rank | Model Architecture | Category | MAE (Wh/m²) | RMSE (Wh/m²) | 95% CI (RMSE) | R² Score | p-value (vs FT-Trans) | Inference Time (ms) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | FT-Transformer | **Deep Learning** | **81.10 ± 7.21** | **110.47 ± 12.04** | **[101.86, 119.08]** | **0.9919 ± 0.0023** | **Baseline Champion** | **0.68** |
| 🥈 | Artificial Neural Network (ANN) | Neural Baseline | 129.13 ± 9.53 | 172.50 ± 14.25 | [162.31, 182.70] | 0.9807 ± 0.0037 | p < 0.001 (***) | 0.18 |
| 🥉 | Histogram Gradient Boosting (HGB) | Ensemble | 129.37 ± 12.36 | 178.39 ± 22.46 | [162.32, 194.46] | 0.9792 ± 0.0055 | p < 0.001 (***) | 0.79 |
| 4 | Support Vector Regression (SVR) | Classical ML | 114.11 ± 9.95 | 188.36 ± 35.15 | [163.22, 213.51] | 0.9765 ± 0.0080 | p < 0.001 (***) | 0.19 |
| 5 | Linear Regression (LR) | Classical ML | 134.28 ± 9.64 | 194.14 ± 42.34 | [163.86, 224.43] | 0.9746 ± 0.0122 | p < 0.001 (***) | 0.07 |
| 6 | XGBoost | Ensemble | 155.65 ± 11.10 | 215.21 ± 20.70 | [200.40, 230.02] | 0.9697 ± 0.0067 | p < 0.001 (***) | 0.33 |
| 7 | Random Forest (RF) | Ensemble | 163.60 ± 17.87 | 230.28 ± 26.87 | [211.06, 249.51] | 0.9653 ± 0.0090 | p < 0.001 (***) | 30.86 |
| 8 | 1D CNN | **Deep Learning** | 214.28 ± 19.76 | 283.96 ± 25.74 | [265.55, 302.37] | 0.9471 ± 0.0133 | p < 0.001 (***) | 1.32 |
| 9 | Solar LSTM | **Deep Learning** | 273.68 ± 40.51 | 362.18 ± 55.91 | [322.18, 402.18] | 0.9117 ± 0.0335 | p < 0.001 (***) | 0.52 |
| 10 | Decision Tree (DT) | Classical ML | 280.33 ± 23.14 | 410.28 ± 51.76 | [373.25, 447.31] | 0.8895 ± 0.0312 | p < 0.001 (***) | 0.08 |
| 11 | K-Nearest Neighbors (KNN) | Classical ML | 337.79 ± 22.74 | 447.20 ± 27.45 | [427.56, 466.84] | 0.8699 ± 0.0236 | p < 0.001 (***) | 2.94 |

> **Statistical Significance & Confidence Analysis:** Paired two-tailed Student's t-tests ($df = 9$) across all 10 cross-validation folds confirm that the performance superiority of **FT-Transformer** over every competing baseline is statistically significant at $p < 0.001\ (***)$. In addition, the 95% Confidence Interval for FT-Transformer's RMSE ($[101.86, 119.08]\text{ Wh/m}^2$) exhibits **zero overlap** with any baseline (nearest being ANN at $[162.31, 182.70]$), demonstrating decisive outperformance. Full t-statistics, p-values, and CIs are exported in [`results/statistical_significance.csv`](results/statistical_significance.csv).

### 🗺️ 43-Fold Leave-One-Station-Out (LOSO) Cross-Validation

Evaluating spatial generalization of the FT-Transformer across all 43 weather stations:

| Metric | Value |
|---|:---:|
| **MAE** | 101.67 ± 30.16 Wh/m² |
| **RMSE** | 132.78 ± 52.61 Wh/m² |
| **R² Score** | 0.9784 ± 0.0381 |

This demonstrates strong generalization to **previously unseen geographic locations** across Saudi Arabia.

---

## 🏗️ Deep Learning Architectures

### 1. FT-Transformer (Feature Tokenizer Transformer) — Champion Model

```
Input: (B, 21 features)
  ├── Numerical Feature Tokenizer: 21 × Linear(1, 64) projections → (B, 21, 64)
  ├── Prepend Learnable [CLS] Token → (B, 22, 64)
  ├── 2× Pre-LN TransformerEncoderLayer (d_model=64, 4 Heads, DimFF=48, Dropout=0.10)
  └── Extract [CLS] Token → LayerNorm → Linear(64, 1) → Output GHI (B, 1)
```

**Key Innovation:** Each numerical feature is independently projected into a learned embedding space via its own linear layer, then processed through multi-head self-attention — enabling the model to learn complex inter-feature interactions (e.g., how DNI modulates with temperature and pressure seasonally).

### 2. Solar 1D CNN

```
Input: (B, 1, 21)
  ├── Stem Conv1d(1, 64) + BatchNorm1d + GELU
  ├── Residual ConvBlock1D(64, 128) + Residual ConvBlock1D(128, 256)
  └── AdaptiveAvgPool1d(1) → Flatten → MLP Head → Output GHI (B, 1)
```

### 3. Solar LSTM

```
Input: (B, 21, 1)
  ├── Linear Feature Projection (1 → 32) + LayerNorm
  ├── 1-Layer Bidirectional LSTM(32, 64, Dropout=0.30)
  └── Global Average Pooling → LayerNorm → MLP Head → Output GHI (B, 1)
```

---

## 🔍 Automated Hyperparameter Tuning (RandomizedSearchCV)

The project includes an automated hyperparameter optimization pipeline using **Scikit-Learn's `RandomizedSearchCV`** paradigm with continuous probability distributions (`scipy.stats.loguniform`, `uniform`, `randint`).

### How It Works

1. **Classical Models (SVR, HGB, XGBoost, RF, ANN)**: Uses `sklearn.model_selection.RandomizedSearchCV` wrapped inside leakage-free `Pipeline` and `TransformedTargetRegressor` estimators evaluated on pre-computed K-Fold CV splits.
2. **Deep Learning Models (Transformer, LSTM, 1D CNN)**: Uses a randomized hyperparameter search loop sampling architecture hyperparameters (learning rate, weight decay, token dimension, attention heads) with full K-Fold cross-validation.
3. Every search run is tracked automatically in **MLflow**.
4. The best configuration is exported to a JSON file in `configs/` for reproducible training.

### Supported Models & Search Spaces

| Model | Tunable Hyperparameters | Distribution / Candidates |
|---|---|---|
| **FT-Transformer** | `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout`, `lr`, `weight_decay`, `batch_size` | Discrete [32, 48, 64], loguniform(1e-4, 5e-3) |
| **LSTM** | `embed_dim`, `hidden_dim`, `num_layers`, `dropout`, `bidirectional`, `lr`, `batch_size` | Discrete [16, 32, 64], loguniform(1e-4, 5e-3) |
| **1D CNN** | `base_channels`, `dropout`, `lr`, `weight_decay`, `batch_size` | Discrete [16, 32, 64], loguniform(1e-4, 5e-3) |
| **SVR** | `C`, `epsilon`, `gamma` | loguniform(1.0, 1000.0), loguniform(0.01, 1.0) |
| **HGB** | `max_depth`, `learning_rate`, `max_iter`, `min_samples_leaf`, `l2_regularization` | randint(3, 11), loguniform(0.01, 0.3) |
| **XGBoost** | `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_alpha` | randint(3, 11), loguniform(0.01, 0.3) |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` | Discrete [50, 100, 150, 200], randint(5, 26) |
| **ANN (MLP)** | `hidden_layer_sizes`, `alpha`, `learning_rate_init` | Layer combinations, loguniform(1e-4, 10.0) |

### Usage

```bash
# Tune FT-Transformer across 10-Fold CV (30 iterations)
python src/tune.py --model transformer --trials 30 --cv-folds 10

# Tune SVR with RandomizedSearchCV (50 iterations)
python src/tune.py --model svr --trials 50 --cv-folds 10

# Retrain with tuned hyperparameters
python src/train.py --model transformer --config configs/transformer_best.json --cv kfold
```

Tuned configurations are saved to `configs/{model_name}_best.json` for reproducibility.

---

## 📦 Data & Model Versioning (DVC)

The project integrates **DVC (Data Version Control)** to define the ML pipeline as a reproducible directed acyclic graph (DAG). Dataset (`dataset.csv`) and model checkpoints (`model/`) remain in Git for immediate reproducibility after cloning.

### DVC Pipeline (`dvc.yaml`)

The complete ML workflow — from hyperparameter tuning through training to evaluation — is codified as a connected DAG with explicit dependency tracking:

```bash
# Visualize pipeline DAG
dvc dag

# Reproduce full pipeline (only re-runs stages whose dependencies changed)
dvc repro

# Reproduce a specific stage
dvc repro train
```

```
+------+     +---------+     +----------+
| tune | --> |  train  | --> | evaluate |
+------+     +---------+     +----------+
```

---

## 📊 MLOps: MLflow Experiment Tracking & Registry Gating

The project uses **MLflow** with an SQLite backend (`sqlite:///mlflow.db`) for experiment tracking, parameter logging, metric recording, and model registry governance.

### Logged Parameters
- **Architecture**: `model_architecture`, `d_model`, `n_heads`, `num_layers`, `hidden_dim`, `total_trainable_parameters`
- **Optimization**: `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `optimizer` (`AdamW`), `scheduler` (`CosineAnnealingLR`), `loss_function` (`MSELoss`)
- **Validation**: `cv_strategy` (`10-fold` or `43-fold LOSO`), `random_seed`, `num_features`

### Logged Metrics & Artifacts
- **Per-Fold & Summary Metrics**: MAE, MSE, RMSE, R², Training Time per fold (Mean ± Std).
- **Artifacts**: Checkpoints (`.pt`), Scaler objects (`.pkl`), Parity scatter plots, Residual error distribution charts.

### Model Registry Promotion Gate (CI/CD)

When a training run finishes:
1. `src/evaluate.py --gate` verifies candidate model metrics against quality thresholds (RMSE ≤ 300, R² ≥ 0.90).
2. Compares candidate RMSE against the current `@champion` model in the MLflow Model Registry.
3. If approved:
   - Registers new version in MLflow Model Registry (`SolarRadiationPredictor`).
   - Assigns the `@champion` model alias.
   - Automatically exports weights and scaler bundle to `model/production_dl_model.pt` and `model/production_dl_scaler.pkl`.
4. If candidate fails: blocks promotion in CI/CD.

```bash
# View all experiments, runs, and registered model versions
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# Open http://127.0.0.1:5001
```

---

## 📁 Project Structure

```
Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset/
│
├── .dvc/                          # DVC configuration and internal metadata
├── .dvcignore                     # DVC ignore rules
├── dvc.yaml                       # Reproducible DVC ML pipeline definition (tune → train → evaluate)
│
├── src/                           # Core source code
│   ├── __init__.py
│   ├── dataset.py                 # Data loading, feature definitions, CV splits
│   ├── train.py                   # MLflow-integrated training pipeline (10-Fold & LOSO CV)
│   ├── tune.py                    # RandomizedSearchCV hyperparameter optimization module
│   ├── evaluate.py                # MLflow Model Registry promotion gate
│   ├── export_results.py          # Granular CSV results exporter (samples, folds, stations)
│   ├── predict.py                 # SolarPredictor inference API
│   └── models/
│       ├── __init__.py            # MODEL_REGISTRY & build_model() factory
│       ├── transformer.py         # FT-Transformer architecture
│       ├── lstm.py                # Solar LSTM architecture
│       └── cnn1d.py               # Solar 1D CNN architecture
│
├── results/                       # Granular, publication-ready research CSV exports
│   ├── benchmark_summary.csv      # Cross-model master comparison summary (with 95% CIs)
│   ├── statistical_significance.csv # Formal hypothesis testing (t-stats, p-values, 95% CIs)
│   ├── paper_baselines.csv        # Published 8-model baseline results
│   ├── *_samples.csv              # Per-observation out-of-fold predictions & residuals
│   ├── *_folds.csv                # Per-fold performance metrics (MAE, RMSE, R², MBE)
│   └── *_stations.csv             # Station-by-station spatial metrics (all 43 stations)
│
├── tests/                         # Test suite (45 tests)
│   ├── __init__.py
│   ├── test_api.py                # Flask API endpoint tests
│   ├── test_models.py             # Forward/backward pass tests for all DL models
│   ├── test_preprocessing.py      # Dataset loading, CV split, and scaling tests
│   ├── test_registry_gate.py      # MLflow promotion gate tests
│   ├── test_results_export.py     # Results export & metrics calculation tests
│   └── test_tuning.py             # RandomizedSearchCV search space & tuning smoke tests
│
├── configs/                       # Tuned hyperparameter configurations (JSON)
│   ├── transformer_best.json      # Best FT-Transformer config (R² = 0.9919)
│   ├── cnn1d_best.json            # Best 1D CNN config (R² = 0.9471)
│   └── lstm_best.json             # Best Solar LSTM config (R² = 0.8930)
│
├── model/                         # Trained model checkpoints & scalers
│   ├── production_dl_model.pt     # Current production champion model
│   ├── production_dl_scaler.pkl   # Production scaler
│   ├── transformer_model.pt       # FT-Transformer checkpoint
│   ├── lstm_model.pt              # LSTM checkpoint
│   ├── cnn1d_model.pt             # 1D CNN checkpoint
│   └── *_classical_model.pkl      # Classical ML model checkpoints
│
├── Other Models/                  # Legacy pre-trained classical models
│
├── app.py                         # Flask web application
├── wsgi.py                        # WSGI entry point for production deployment
├── templates/
│   └── index.html                 # Web app frontend (Leaflet.js, Plotly.js)
├── static/
│   └── app.js                     # Frontend JavaScript logic
│
├── dataset.csv                    # Saudi Arabia solar radiation dataset (1,649 records)
├── requirements.txt               # Production Python dependencies (Flask, PyTorch, sklearn)
├── requirements-dev.txt           # Development dependencies (MLflow, DVC, scipy, pytest)
├── Dockerfile                     # Docker container configuration
├── .dockerignore                  # Docker build exclusions
├── .github/workflows/
│   ├── ci.yml                     # Continuous Integration & MLflow promotion gate
│   └── cd.yml                     # Continuous Deployment (Docker build & webhook)
│
├── Contribute.md                  # Contributing guidelines
├── Learn.md                       # Learning guide for newcomers
├── LICENSE                        # MIT License
└── README.md                      # This document
```

---

## 🚀 Quickstart Guide

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/Aamir-Sarang31/Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset.git
cd Solar-Radiation-Prediction-using-Saudi-Arabia-Dataset

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install all dependencies (production + training/MLOps tools)
pip install -r requirements-dev.txt
```

### 2. Train Models

```bash
# Train all 11 models (classical + deep learning) with 10-Fold CV
python src/train.py --model all --epochs 25 --cv kfold

# Train only the FT-Transformer
python src/train.py --model transformer --epochs 25 --cv kfold

# Train with 43-Fold Leave-One-Station-Out CV
python src/train.py --model transformer --epochs 25 --cv loso

# Train with tuned hyperparameters
python src/train.py --model transformer --config configs/transformer_best.json --cv kfold
```

### 3. Run Hyperparameter Tuning

```bash
# Tune FT-Transformer (30 iterations, 10-Fold CV)
python src/tune.py --model transformer --trials 30 --cv-folds 10

# Tune SVR with RandomizedSearchCV (50 iterations)
python src/tune.py --model svr --trials 50

# Tune all supported models
python src/tune.py --model hgb --trials 50
python src/tune.py --model xgb --trials 50
```

### 4. Run Tests & Promotion Gate

```bash
# Run complete test suite (41 tests)
python -m pytest tests/ -v

# Run MLflow Model Registry Promotion Gate
python src/evaluate.py --gate --model transformer \
    --checkpoint model/transformer_model.pt \
    --scaler model/transformer_scaler.pkl
```

### 5. Launch Web Application

```bash
python app.py
# Open http://localhost:5000
```

### 6. View MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

---

## 🌐 Web Application

The Flask web application provides:

- **Interactive Leaflet.js Station Map**: Displays all 43 weather stations across Saudi Arabia with GHI heatmap overlays
- **Multi-Model GHI Predictor**: Switch dynamically between Linear Regression, ANN, FT-Transformer, LSTM, and 1D CNN
- **Station & Feature Comparisons**: Interactive Plotly.js visualizations
- **Real-Time Predictions**: Input meteorological features and get instant GHI forecasts

```bash
# Run locally
python app.py

# Production deployment (Gunicorn / Waitress)
gunicorn wsgi:app --bind 0.0.0.0:8000    # Linux/macOS
waitress-serve --port=8000 wsgi:app       # Windows
```

---

## 📋 Dataset

| Property | Value |
|---|---|
| **Source** | King Fahd University of Petroleum & Minerals (KFUPM) Solar Radiation Monitoring Network |
| **Coverage** | 43 weather stations across Saudi Arabia |
| **Period** | 2017–2021 |
| **Records** | 1,649 monthly aggregated observations |
| **Target Variable** | Global Horizontal Irradiance (GHI) in Wh/m² |
| **Features (21)** | Temperature (avg/max/min), Humidity (avg/max/min), Pressure, Wind Speed/Direction, DNI, DHI, Sunshine Duration, Rainfall, Latitude, Longitude, Elevation, and temporal features |

---

## 🔄 CI/CD Workflows

### `.github/workflows/ci.yml` — Continuous Integration

Triggers on every push and pull request to `main`/`master`:

1. **Checkout** → Set up Python 3.11
2. **Install Dependencies** → PyTorch (CPU) + `requirements-dev.txt` (MLflow, DVC, scipy, pytest, flake8)
3. **Code Quality** → Flake8 linting (syntax errors, undefined names)
4. **Test Suite** → `pytest tests/ -v` (46 tests)
5. **Smoke Training** → 3-epoch FT-Transformer training with MLflow tracking
6. **Promotion Gate** → MLflow Registry quality threshold verification (`--dry-run` on PRs, full export on `main`)

### `.github/workflows/cd.yml` — Continuous Deployment

Triggers automatically on push to `main`/`master`:

1. **Docker Buildx** → Builds containerized production image using `Dockerfile`
2. **Deployment Trigger** → Invokes automated webhook for production hosting (e.g., Render, Railway)

---

## 🧪 Testing

The project includes **46 automated tests** across 6 test modules:

| Module | Tests | Coverage |
|---|:---:|---|
| `test_api.py` | 7 | Flask route responses, prediction endpoints, multi-model inference |
| `test_models.py` | 7 | Forward pass, backward pass gradients, single-sample inference for all DL models |
| `test_preprocessing.py` | 5 | Feature list completeness, dataset loading, 10-Fold/LOSO splits, scaling |
| `test_registry_gate.py` | 3 | Promotion thresholds, acceptance criteria, MLflow retrieval |
| `test_results_export.py` | 5 | Research CSV results export, schema validation, 95% CIs, statistical significance (p-values) |
| `test_tuning.py` | 19 | RandomizedSearchCV distributions, sampling (8 models), device detection, classical/DL smoke tuning |

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_tuning.py -v
```

---

## 👥 Authors & Acknowledgements

- **Aamir Sarang** ([@Aamir-Sarang31](https://github.com/Aamir-Sarang31))
- **Nishant Narudkar** ([@nishnarudkar](https://github.com/nishnarudkar))
- **Maitreya Pawar** ([@Metzo64](https://github.com/Metzo64))
- **Vatsal Parmar** ([@Vatsal211005](https://github.com/Vatsal211005))

We express our sincere gratitude to **Mr. Pramod H. Kachare** and **Mr. Sandeep Sangle** for their valuable guidance and mentorship throughout this research project.

---

## 📄 License

This project is open-source under the **MIT License**. See [LICENSE](LICENSE) for details.
