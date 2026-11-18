"""
Dataset loading, preprocessing, scaling, and cross-validation partitioners.
Aligned with Springer Nature research paper specifications (10-fold CV and 43-fold LOSO CV).

Feature Selection Rationale
============================

RADIOMETRIC INPUTS EXCLUDED (target leakage):
----------------------------------------------
The following 8 columns were present in the raw KFUPM dataset but are deliberately
excluded from model inputs because they are algebraically derived from, or statistically
co-measured with, the prediction target (GHI):

  • DHI (Wh/m2)                     — Diffuse Horizontal Irradiance. Together with DNI,
                                       it determines GHI via the closure equation:
                                       GHI ≈ DNI·cos(θz) + DHI. Including either
                                       component gives any linear model near-perfect R².
  • DHI Uncertainty (Wh/m2)         — Instrument uncertainty estimated *from* DHI
                                       measurements; its magnitude is proportional to DHI.
  • Standard Deviation DHI (Wh/m2)  — Temporal variability of DHI; directly coupled
                                       to DHI and hence to GHI.
  • DNI (Wh/m2)                     — Direct Normal Irradiance. The other term in the
                                       GHI closure equation; see DHI note above.
  • DNI Uncertainty (Wh/m2)         — Same reasoning as DHI Uncertainty.
  • Standard Deviation DNI (Wh/m2)  — Same reasoning as Standard Deviation DHI.
  • GHI Uncertainty (Wh/m2)         — Instrument uncertainty reported *for the target
                                       variable itself*; it encodes the magnitude of GHI.
  • Standard Deviation GHI (Wh/m2)  — Temporal spread of GHI within the aggregation
                                       window; trivially correlated with GHI.

Including any of these would constitute target leakage: even a linear model trivially
reaches R² ≈ 0.97–0.99 because it is effectively reconstructing GHI from GHI. All
leakage-free results reported in this project use only the 13 features below.

RATIONALE FOR NON-INCLUSION OF CANDIDATE LITERATURE FEATURES:
---------------------------------------------------------------------------
In the broader solar irradiance literature, several features are sometimes cited as
predictors: Latitude, Longitude, Elevation, Sunshine Duration, Rainfall, and Max/Min
Temperature and Humidity. None of these are used as input features in our model suite.
The formal physical, methodological, and sensor-network reasons are as follows:

  1. Latitude & Longitude (Spatial Coordinates):
     - Status: Present in dataset.csv as station metadata.
     - Decision: Deliberately excluded from the model feature vector.
     - Reasoning: Coordinates are static, time-invariant station identifiers. Including
       (Lat, Lon) in tabular ML encourages decision trees and neural networks to memorize
       specific station locations (acting as a spatial lookup table) rather than learning
       the genuine atmospheric relationships between weather dynamics and irradiance.
       Crucially, in 43-Fold Leave-One-Station-Out (LOSO) cross-validation, models evaluated
       on unseen geographical coordinates cannot rely on spatial memorization. Excluding
       coordinates forces models to generalize purely based on universal atmospheric physics.
       Coordinates are strictly reserved for mapping, geospatial clustering, and fold grouping.

  2. Elevation:
     - Status: Not recorded as a separate column in dataset.csv.
     - Decision: Excluded; physically subsumed by Barometric Pressure.
     - Reasoning: Barometric Pressure (mB) is measured at every station and acts as a direct,
       dynamic physical surrogate for elevation via the barometric formula:
       P = P_0 * exp(-M*g*h / (R*T)).
       Unlike static topographic elevation, local barometric pressure accounts for both station
       altitude and dynamic synoptic weather patterns (high/low pressure systems) that govern
       air column density and atmospheric optical mass. Adding static elevation would be
       redundant and physically inferior to real-time atmospheric pressure.

  3. Sunshine Duration:
     - Status: Absent from dataset.csv.
     - Decision: Excluded due to sensor network scope and target leakage prevention.
     - Reasoning: Automated K.A.CARE / KFUPM research stations use high-precision thermopile
       radiometers and ultrasonic weather sensors rather than Campbell-Stokes glass spheres or
       optical sunshine recorders. Furthermore, WMO defines Sunshine Duration as the time during
       which Direct Normal Irradiance (DNI) exceeds 120 W/m². Sunshine duration is therefore a
       derived radiometric proxy of DNI/GHI; including it would reintroduce target leakage.

  4. Rainfall / Precipitation:
     - Status: Absent from dataset.csv.
     - Decision: Excluded due to arid-zone monitoring station design.
     - Reasoning: The monitoring network covers hyper-arid and arid climate zones in the
       Arabian Peninsula where annual precipitation is extremely low (<50-100 mm/year) and
       highly episodic. Solar research stations in hyper-arid zones prioritize radiometric,
       thermal, and aeolian sensors over rain gauges. On a monthly timescale in desert regimes,
       rainfall has negligible predictive variance compared to aerosol optical depth and clouds.

  5. Max / Min Temperature and Max / Min Humidity:
     - Status: Discrete single-instant monthly extremes are absent in dataset.csv.
     - Decision: Subsumed by continuous Mean, Standard Deviation, and Uncertainty.
     - Reasoning: The dataset provides monthly aggregated records with continuous mean values
       and their monthly standard deviations (e.g., 'Standard Deviation Air Temperature',
       'Standard Deviation Relative Humidity'). In atmospheric modeling, single-point extremes
       (max/min) are highly susceptible to sensor noise, thermal micro-spikes, and localized
       artifacts. In contrast, standard deviation captures the full diurnal thermal amplitude
       and moisture volatility across all 30 days of the month with far greater statistical
       robustness. Diurnal temperature spread directly reflects atmospheric transmissivity
       (clear skies produce wide diurnal temperature swings, while dust/cloud cover dampens variance).

These exclusions ensure our 13 features represent a clean, physically sound, non-leaking
meteorological observation suite.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Generator, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, LeaveOneGroupOut

DEG = chr(176)

# 13 clean meteorological features — radiometric proxies of GHI (DNI, DHI and
# their uncertainties/std devs, plus GHI Uncertainty and Standard Deviation GHI)
# have been removed to eliminate target leakage. See module docstring above for
# the full justification.
FEATURE_NAMES = [
    f"Air Temperature (C{DEG})",
    f"Air Temperature Uncertainty (C{DEG})",
    f"Wind Direction at 3m ({DEG}N)",
    f"Wind Direction at 3m Uncertainty ({DEG}N)",
    "Wind Speed at 3m (m/s)",
    "Wind Speed at 3m Uncertainty (m/s)",
    "Wind Speed at 3m (std dev) (m/s)",
    "Peak Wind Speed at 3m (m/s)",
    "Peak Wind Speed at 3m Uncertainty (m/s)",
    "Relative Humidity (%)",
    "Relative Humidity Uncertainty (%)",
    "Barometric Pressure (mB (hPa equiv))",
    "Barometric Pressure Uncertainty (mB (hPa equiv))"
]

TARGET_NAME = "GHI (Wh/m2)"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any non-standard degree symbols or double-encoded characters in headers."""
    col_map = {}
    for col in df.columns:
        clean = col.replace(chr(194) + chr(176), chr(176)).replace("Â°", chr(176))
        col_map[col] = clean
    return df.rename(columns=col_map)


def load_dataset(csv_path: str = "dataset.csv") -> pd.DataFrame:
    """
    Load and clean dataset from CSV.
    Imputes missing values with feature median as specified in Section 3.2 of the paper.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin-1")

    df = clean_column_names(df)

    # Impute missing values for numerical features using column median
    for col in FEATURE_NAMES:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Drop rows missing the target variable if any
    if TARGET_NAME in df.columns:
        df = df.dropna(subset=[TARGET_NAME])

    return df


class SolarDataset(Dataset):
    """PyTorch Dataset for solar radiation feature rows."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_10_fold_cv_splits(
    df: pd.DataFrame,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate 10-fold cross-validation index splits."""
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    return list(kf.split(df))


def get_loso_cv_splits(
    df: pd.DataFrame
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Generate Leave-One-Station-Out (LOSO) 43-fold cross-validation splits.
    Yields (train_indices, test_indices, excluded_station_name).
    """
    logo = LeaveOneGroupOut()
    groups = df["Station_Name"].values
    splits = []
    for train_idx, test_idx in logo.split(df, groups=groups):
        station_name = df.iloc[test_idx[0]]["Station_Name"]
        splits.append((train_idx, test_idx, station_name))
    return splits


def prepare_fold_data(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scaler_type: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Extract, scale, and return train/test arrays and the fitted scaler.
    Ensures scaler is fit strictly on training data to eliminate leakage.
    """
    X_train_raw = df.iloc[train_idx][FEATURE_NAMES].values
    y_train = df.iloc[train_idx][TARGET_NAME].values
    X_test_raw = df.iloc[test_idx][FEATURE_NAMES].values
    y_test = df.iloc[test_idx][TARGET_NAME].values

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, y_train, X_test, y_test, scaler
