"""
Optuna-based Automated Hyperparameter Optimization Pipeline with MLflow Integration.
Supports 10-Fold CV (or custom folds) across Deep Learning and Classical ML architectures.
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, Tuple, Optional

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import mlflow

from src.dataset import (
    load_dataset,
    get_10_fold_cv_splits,
    SolarDataset,
    FEATURE_NAMES,
    TARGET_NAME
)
from src.models import build_model


def get_default_device(requested_device: Optional[str] = None) -> str:
    """Return requested device or automatically detect CUDA GPU availability."""
    if requested_device:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_fold_data_scaled(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scaler_type: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    """Prepare scaled train/test splits with zero data leakage."""
    X_train_raw = df.iloc[train_idx][FEATURE_NAMES].values
    y_train_raw = df.iloc[train_idx][TARGET_NAME].values
    X_test_raw = df.iloc[test_idx][FEATURE_NAMES].values
    y_test_raw = df.iloc[test_idx][TARGET_NAME].values

    if scaler_type == "minmax":
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    else:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1))
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1))

    return X_train, y_train, X_test, y_test, x_scaler, y_scaler


def sample_hyperparameters(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """Sample candidate hyperparameters for the specified model from its search space."""
    model_key = model_name.lower()

    if model_key in ["transformer", "ft_transformer"]:
        d_model = trial.suggest_categorical("d_model", [32, 48, 64])
        # nhead must divide d_model
        valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        nhead = trial.suggest_categorical("nhead", valid_heads)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dim_feedforward = trial.suggest_int("dim_feedforward", 48, 192, step=16)
        dropout = trial.suggest_float("dropout", 0.05, 0.3, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        return {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size
        }

    elif model_key == "lstm":
        embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 64])
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        return {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size
        }

    elif model_key == "cnn1d":
        base_channels = trial.suggest_categorical("base_channels", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        return {
            "base_channels": base_channels,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size
        }

    elif model_key == "svr":
        C = trial.suggest_float("C", 1.0, 1000.0, log=True)
        epsilon = trial.suggest_float("epsilon", 0.01, 1.0, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

        return {
            "C": C,
            "epsilon": epsilon,
            "gamma": gamma
        }

    elif model_key == "hgb":
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_iter = trial.suggest_int("max_iter", 50, 300, step=50)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 50)
        l2_regularization = trial.suggest_float("l2_regularization", 0.0, 10.0)

        return {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization
        }

    elif model_key == "xgb":
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)

        return {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha
        }

    elif model_key == "rf":
        n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
        max_depth = trial.suggest_int("max_depth", 5, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }

    elif model_key == "ann":
        layer1 = trial.suggest_categorical("hidden_layer_1", [16, 32, 64, 128])
        layer2 = trial.suggest_categorical("hidden_layer_2", [0, 8, 16, 32, 64])
        hidden_layer_sizes = (layer1,) if layer2 == 0 else (layer1, layer2)
        alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)

        return {
            "hidden_layer_sizes": hidden_layer_sizes,
            "alpha": alpha,
            "learning_rate_init": learning_rate_init
        }

    else:
        raise ValueError(f"Tuning not supported for model: '{model_name}'")


def train_and_eval_dl_fold(
    model_name: str,
    params: Dict[str, Any],
    train_loader: DataLoader,
    test_loader: DataLoader,
    y_test_raw: np.ndarray,
    y_scaler: object,
    epochs: int = 20,
    device: str = "cpu"
) -> float:
    """Train DL model on one fold with given params and return fold RMSE."""
    # Filter architecture kwargs for build_model
    arch_kwargs = {
        k: v for k, v in params.items()
        if k not in ["lr", "weight_decay", "batch_size"]
    }
    model = build_model(model_name, **arch_kwargs).to(device)

    lr = params.get("lr", 1e-3)
    weight_decay = params.get("weight_decay", 1e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    # Inference on test set
    model.eval()
    all_preds_scaled = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            all_preds_scaled.extend(preds.cpu().numpy().flatten())

    y_pred_scaled = np.array(all_preds_scaled).reshape(-1, 1)
    y_pred_raw = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_pred_raw = np.clip(y_pred_raw, 0.0, None)

    return float(np.sqrt(mean_squared_error(y_test_raw, y_pred_raw)))


def train_and_eval_classical_fold(
    model_name: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train_raw: np.ndarray,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    seed: int = 42
) -> float:
    """Train classical model on one fold with given params and return fold RMSE."""
    model_key = model_name.lower()

    if model_key == "svr":
        model = TransformedTargetRegressor(
            regressor=SVR(
                C=params["C"],
                epsilon=params["epsilon"],
                gamma=params["gamma"]
            ),
            transformer=StandardScaler()
        )
    elif model_key == "hgb":
        model = HistGradientBoostingRegressor(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            random_state=seed
        )
    elif model_key == "xgb":
        if HAS_XGB:
            model = XGBRegressor(
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_alpha=params["reg_alpha"],
                random_state=seed
            )
        else:
            model = GradientBoostingRegressor(
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                subsample=params["subsample"],
                random_state=seed
            )
    elif model_key == "rf":
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1
        )
    elif model_key == "ann":
        model = TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=params["alpha"],
                learning_rate_init=params["learning_rate_init"],
                max_iter=800,
                early_stopping=True,
                random_state=seed
            ),
            transformer=StandardScaler()
        )
    else:
        raise ValueError(f"Unsupported classical model for tuning: {model_name}")

    model.fit(X_train, y_train_raw)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, None)

    return float(np.sqrt(mean_squared_error(y_test_raw, y_pred)))


def create_objective(
    model_name: str,
    df: pd.DataFrame,
    cv_splits: list,
    epochs: int = 20,
    device: str = "cpu",
    seed: int = 42
):
    """Factory returning an Optuna objective function for K-Fold CV optimization."""
    is_dl = model_name.lower() in ["transformer", "ft_transformer", "lstm", "cnn1d"]

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial, model_name)
        batch_size = params.get("batch_size", 32)
        fold_rmses = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train, y_train, X_test, y_test, x_scaler, y_scaler = prepare_fold_data_scaled(
                df, train_idx, test_idx, scaler_type="standard"
            )
            y_test_raw = df.iloc[test_idx][TARGET_NAME].values
            y_train_raw = df.iloc[train_idx][TARGET_NAME].values

            if is_dl:
                train_dataset = SolarDataset(X_train, y_train)
                test_dataset = SolarDataset(X_test, y_test)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                fold_rmse = train_and_eval_dl_fold(
                    model_name=model_name,
                    params=params,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    y_test_raw=y_test_raw,
                    y_scaler=y_scaler,
                    epochs=epochs,
                    device=device
                )
            else:
                fold_rmse = train_and_eval_classical_fold(
                    model_name=model_name,
                    params=params,
                    X_train=X_train,
                    y_train_raw=y_train_raw,
                    X_test=X_test,
                    y_test_raw=y_test_raw,
                    seed=seed
                )

            fold_rmses.append(fold_rmse)

            # Report intermediate running mean RMSE for trial pruning
            intermediate_mean = float(np.mean(fold_rmses))
            trial.report(intermediate_mean, step=fold_idx)

            # Check if this trial should be pruned early
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_rmses))

    return objective


def run_tuning(
    model_name: str,
    n_trials: int = 30,
    cv_folds: int = 10,
    epochs: int = 20,
    device: Optional[str] = None,
    dataset_path: str = "dataset.csv",
    output_dir: str = "configs",
    seed: int = 42
) -> Tuple[Dict[str, Any], float]:
    """
    Execute hyperparameter search with Optuna and log results to MLflow.
    """
    target_device = get_default_device(device)
    print(f"\n==================================================")
    print(f" Optuna Hyperparameter Tuning: {model_name.upper()}")
    print(f" Trials: {n_trials} | CV Strategy: {cv_folds}-Fold CV | Device: {target_device.upper()}")
    print(f"==================================================")

    df = load_dataset(dataset_path)
    cv_splits = get_10_fold_cv_splits(df, seed=seed)
    if cv_folds < len(cv_splits):
        cv_splits = cv_splits[:cv_folds]

    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Solar_Radiation_Prediction")

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study_name = f"tune_{model_name}_{int(time.time())}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )

    objective_fn = create_objective(
        model_name=model_name,
        df=df,
        cv_splits=cv_splits,
        epochs=epochs,
        device=target_device,
        seed=seed
    )

    with mlflow.start_run(run_name=f"optuna_tune_{model_name}"):
        mlflow.log_params({
            "tuning_model": model_name,
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "epochs_per_trial": epochs,
            "device": target_device,
            "seed": seed
        })

        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_rmse = study.best_value

        print(f"\n==================== TUNING COMPLETED ====================")
        print(f" Best {cv_folds}-Fold Mean RMSE: {best_rmse:.2f} Wh/m²")
        print(f" Best Hyperparameters:")
        for k, v in best_params.items():
            print(f"   -> {k}: {v}")
        print(f"==========================================================\n")

        # Log best results to MLflow
        mlflow.log_metric("best_cv_rmse", best_rmse)
        for k, v in best_params.items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(f"best_{k}", v)

        # Export best config to JSON
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, f"{model_name.lower()}_best.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)

        mlflow.log_artifact(config_path, artifact_path="tuned_configs")
        print(f" Exported best configuration to: {config_path}")

    return best_params, best_rmse


def main():
    parser = argparse.ArgumentParser(description="Automated Hyperparameter Optimization via Optuna")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["transformer", "ft_transformer", "lstm", "cnn1d", "svr", "hgb", "xgb", "rf", "ann"],
        help="Architecture to tune"
    )
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--cv-folds", type=int, default=10, help="Number of CV folds to evaluate per trial")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per fold for DL models")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--dataset", type=str, default="dataset.csv", help="Path to dataset.csv")
    parser.add_argument("--output-dir", type=str, default="configs", help="Directory to save best configs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_tuning(
        model_name=args.model,
        n_trials=args.trials,
        cv_folds=args.cv_folds,
        epochs=args.epochs,
        device=args.device,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
