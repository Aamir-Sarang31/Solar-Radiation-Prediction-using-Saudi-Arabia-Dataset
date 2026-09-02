"""
Comprehensive Results Exporter for Solar Radiation Prediction Research.

Exports granular, publication-ready CSV files:
1. Sample-level out-of-fold predictions with residual and meteorological variables.
2. Per-fold metrics table (MAE, MSE, RMSE, R2, MBE, NMAE%, NRMSE%, time).
3. Per-station aggregated metrics across all 43 monitoring stations.
4. Master benchmark comparison summary across all evaluated models.
"""

import os
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute full suite of regression metrics standard in solar irradiance literature:
    MAE, MSE, RMSE, R2, MBE (Mean Bias Error), NMAE (%), and NRMSE (%).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 1e-9 else 0.0
    mbe = float(np.mean(y_pred - y_true))

    mean_true = float(np.mean(y_true))
    nmae_pct = float((mae / mean_true) * 100.0) if mean_true > 1e-6 else 0.0
    nrmse_pct = float((rmse / mean_true) * 100.0) if mean_true > 1e-6 else 0.0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mbe": mbe,
        "nmae_pct": nmae_pct,
        "nrmse_pct": nrmse_pct
    }


def get_model_category(model_name: str) -> str:
    """Return academic categorization for model name."""
    name = model_name.lower()
    if name in ["transformer", "ft_transformer", "lstm", "cnn1d"]:
        return "Deep Learning"
    elif name in ["ann"]:
        return "Neural Baseline"
    elif name in ["hgb", "xgb", "rf"]:
        return "Ensemble"
    else:
        return "Classical ML"


def export_cross_validation_csvs(
    df: pd.DataFrame,
    fold_splits: list,
    fold_predictions_list: List[np.ndarray],
    fold_metrics_list: List[Dict[str, float]],
    model_name: str,
    cv_strategy: str,
    results_dir: str = "results",
    target_col: str = "GHI (Wh/m2)"
) -> Dict[str, str]:
    """
    Build and save sample-level, fold-level, station-level, and master summary CSVs.

    Args:
        df: Original cleaned DataFrame with features, metadata, and target.
        fold_splits: List of split tuples (train_idx, test_idx) or (train_idx, test_idx, station_name).
        fold_predictions_list: List of prediction arrays, one per fold.
        fold_metrics_list: List of metric dicts, one per fold.
        model_name: Name of evaluated model.
        cv_strategy: 'kfold' or 'loso'.
        results_dir: Directory where CSV files are saved.
        target_col: Target column name.

    Returns:
        Dictionary containing filepaths of the 4 generated CSV files.
    """
    os.makedirs(results_dir, exist_ok=True)
    sample_records = []
    fold_records = []

    # Map candidate meteorological column names
    temp_col = next((c for c in df.columns if "Air Temperature" in c and "Uncertainty" not in c), None)
    dhi_col = next((c for c in df.columns if c.startswith("DHI") and "Uncertainty" not in c and "Standard" not in c), None)
    dni_col = next((c for c in df.columns if c.startswith("DNI") and "Uncertainty" not in c and "Standard" not in c), None)
    wind_col = next((c for c in df.columns if "Wind Speed at 3m" in c and "Uncertainty" not in c and "std" not in c and "Peak" not in c), None)
    rh_col = next((c for c in df.columns if "Relative Humidity" in c and "Uncertainty" not in c), None)
    press_col = next((c for c in df.columns if "Barometric Pressure" in c and "Uncertainty" not in c), None)

    for fold_idx, split_info in enumerate(fold_splits):
        if cv_strategy == "loso":
            train_idx, test_idx, station_name_loso = split_info
            held_out_station = str(station_name_loso)
        else:
            train_idx, test_idx = split_info
            held_out_station = "Multiple"

        y_pred = fold_predictions_list[fold_idx]
        metrics = fold_metrics_list[fold_idx]

        test_sub_df = df.iloc[test_idx]
        y_true = test_sub_df[target_col].values

        # 1. Collect per-fold record
        fold_rec = {
            "fold": fold_idx + 1,
            "model_name": model_name,
            "category": get_model_category(model_name),
            "cv_strategy": cv_strategy,
            "held_out_station": held_out_station,
            "num_train_samples": len(train_idx),
            "num_test_samples": len(test_idx),
            "actual_mean_ghi": float(np.mean(y_true)),
            "actual_std_ghi": float(np.std(y_true)),
            "pred_mean_ghi": float(np.mean(y_pred)),
            "pred_std_ghi": float(np.std(y_pred)),
            "mae": float(metrics["mae"]),
            "mse": float(metrics["mse"]),
            "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]),
            "mbe": float(metrics.get("mbe", np.mean(y_pred - y_true))),
            "nmae_pct": float(metrics.get("nmae_pct", (metrics["mae"] / np.mean(y_true)) * 100.0)),
            "nrmse_pct": float(metrics.get("nrmse_pct", (metrics["rmse"] / np.mean(y_true)) * 100.0)),
            "training_time_s": float(metrics.get("training_time", 0.0))
        }
        fold_records.append(fold_rec)

        # 2. Collect sample-level records
        for i, global_idx in enumerate(test_idx):
            row = test_sub_df.iloc[i]
            y_t = float(y_true[i])
            y_p = float(y_pred[i])
            res_err = float(y_p - y_t)
            abs_err = float(abs(res_err))
            sq_err = float(res_err ** 2)
            rel_err = float((abs_err / max(abs(y_t), 1e-3)) * 100.0)

            dni_val = float(row[dni_col]) if dni_col and pd.notnull(row[dni_col]) else np.nan
            dhi_val = float(row[dhi_col]) if dhi_col and pd.notnull(row[dhi_col]) else np.nan
            diffuse_fraction = float(dhi_val / (dni_val + dhi_val)) if (dni_val + dhi_val) > 1e-3 else np.nan

            s_rec = {
                "record_id": int(global_idx),
                "fold": fold_idx + 1,
                "model_name": model_name,
                "cv_strategy": cv_strategy,
                "station_name": str(row.get("Station_Name", held_out_station)),
                "latitude": float(row.get("Latitude", np.nan)) if "Latitude" in row else np.nan,
                "longitude": float(row.get("Longitude", np.nan)) if "Longitude" in row else np.nan,
                "date": str(row.get("Date", "")),
                "actual_ghi": y_t,
                "predicted_ghi": y_p,
                "residual_error": res_err,
                "absolute_error": abs_err,
                "squared_error": sq_err,
                "relative_error_pct": rel_err,
                "air_temperature_c": float(row[temp_col]) if temp_col and pd.notnull(row[temp_col]) else np.nan,
                "wind_speed_m_s": float(row[wind_col]) if wind_col and pd.notnull(row[wind_col]) else np.nan,
                "dhi_wh_m2": dhi_val,
                "dni_wh_m2": dni_val,
                "diffuse_fraction": diffuse_fraction,
                "relative_humidity_pct": float(row[rh_col]) if rh_col and pd.notnull(row[rh_col]) else np.nan,
                "barometric_pressure_mb": float(row[press_col]) if press_col and pd.notnull(row[press_col]) else np.nan
            }
            sample_records.append(s_rec)

    # -----------------------------
    # 1. Export Samples CSV
    # -----------------------------
    samples_df = pd.DataFrame(sample_records)
    samples_csv_path = os.path.join(results_dir, f"{model_name}_{cv_strategy}_samples.csv")
    samples_df.to_csv(samples_csv_path, index=False)

    # -----------------------------
    # 2. Export Folds CSV
    # -----------------------------
    folds_df = pd.DataFrame(fold_records)
    folds_csv_path = os.path.join(results_dir, f"{model_name}_{cv_strategy}_folds.csv")
    folds_df.to_csv(folds_csv_path, index=False)

    # -----------------------------
    # 3. Export Stations CSV
    # -----------------------------
    station_records = []
    grouped_stations = samples_df.groupby("station_name")
    for st_name, st_group in grouped_stations:
        st_actual = st_group["actual_ghi"].values
        st_pred = st_group["predicted_ghi"].values
        st_metrics = compute_comprehensive_metrics(st_actual, st_pred)

        lat = st_group["latitude"].dropna().iloc[0] if not st_group["latitude"].dropna().empty else np.nan
        lon = st_group["longitude"].dropna().iloc[0] if not st_group["longitude"].dropna().empty else np.nan

        station_records.append({
            "station_name": st_name,
            "latitude": lat,
            "longitude": lon,
            "model_name": model_name,
            "cv_strategy": cv_strategy,
            "num_samples": len(st_group),
            "actual_mean_ghi": float(np.mean(st_actual)),
            "actual_std_ghi": float(np.std(st_actual)),
            "pred_mean_ghi": float(np.mean(st_pred)),
            "pred_std_ghi": float(np.std(st_pred)),
            "mae": st_metrics["mae"],
            "mse": st_metrics["mse"],
            "rmse": st_metrics["rmse"],
            "r2": st_metrics["r2"],
            "mbe": st_metrics["mbe"],
            "nmae_pct": st_metrics["nmae_pct"],
            "nrmse_pct": st_metrics["nrmse_pct"],
            "max_absolute_error": float(st_group["absolute_error"].max())
        })

    stations_df = pd.DataFrame(station_records).sort_values(by="mae")
    stations_csv_path = os.path.join(results_dir, f"{model_name}_{cv_strategy}_stations.csv")
    stations_df.to_csv(stations_csv_path, index=False)

    # -----------------------------
    # 4. Update Benchmark Summary CSV
    # -----------------------------
    summary_csv_path = os.path.join(results_dir, "benchmark_summary.csv")
    new_summary_row = {
        "model_name": model_name,
        "category": get_model_category(model_name),
        "cv_strategy": cv_strategy,
        "num_folds": len(fold_records),
        "total_test_samples": len(samples_df),
        "mean_mae": float(folds_df["mae"].mean()),
        "std_mae": float(folds_df["mae"].std()),
        "mean_rmse": float(folds_df["rmse"].mean()),
        "std_rmse": float(folds_df["rmse"].std()),
        "mean_r2": float(folds_df["r2"].mean()),
        "std_r2": float(folds_df["r2"].std()),
        "mean_mbe": float(folds_df["mbe"].mean()),
        "std_mbe": float(folds_df["mbe"].std()),
        "mean_nmae_pct": float(folds_df["nmae_pct"].mean()),
        "std_nmae_pct": float(folds_df["nmae_pct"].std()),
        "mean_nrmse_pct": float(folds_df["nrmse_pct"].mean()),
        "std_nrmse_pct": float(folds_df["nrmse_pct"].std()),
        "avg_train_time_s": float(folds_df["training_time_s"].mean()),
        "total_train_time_s": float(folds_df["training_time_s"].sum()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(summary_csv_path):
        summary_df = pd.read_csv(summary_csv_path)
        # Replace if model and cv_strategy match, else append
        match_mask = (summary_df["model_name"] == model_name) & (summary_df["cv_strategy"] == cv_strategy)
        if match_mask.any():
            summary_df = summary_df[~match_mask]
        summary_df = pd.concat([summary_df, pd.DataFrame([new_summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([new_summary_row])

    summary_df = summary_df.sort_values(by=["cv_strategy", "mean_rmse"])
    summary_df.to_csv(summary_csv_path, index=False)

    print(f" [CSV EXPORT COMPLETE] Saved comprehensive results to '{results_dir}/':")
    print(f"   -> Samples CSV:   {samples_csv_path} ({len(samples_df)} records)")
    print(f"   -> Folds CSV:     {folds_csv_path} ({len(folds_df)} folds)")
    print(f"   -> Stations CSV:  {stations_csv_path} ({len(stations_df)} stations)")
    print(f"   -> Benchmark CSV: {summary_csv_path}")

    return {
        "samples_csv": samples_csv_path,
        "folds_csv": folds_csv_path,
        "stations_csv": stations_csv_path,
        "summary_csv": summary_csv_path
    }
