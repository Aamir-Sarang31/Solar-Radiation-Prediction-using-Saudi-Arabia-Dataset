"""
Unit tests for MLflow Model Registry promotion gate logic.
"""

import os
import pytest
from src.evaluate import check_promotion_gate, TARGET_R2_THRESHOLD, MAX_RMSE_THRESHOLD


def test_promotion_gate_thresholds():
    """Verify promotion gate rejects models with bad R2 or RMSE."""
    # Bad R2
    assert not check_promotion_gate(
        candidate_model_name="bad_model",
        candidate_rmse=180.0,
        candidate_r2=0.80,  # Below TARGET_R2_THRESHOLD
        candidate_checkpoint="dummy.pt",
        candidate_scaler="dummy.pkl"
    )

    # Bad RMSE
    assert not check_promotion_gate(
        candidate_model_name="bad_model",
        candidate_rmse=450.0,  # Above MAX_RMSE_THRESHOLD
        candidate_r2=0.96,
        candidate_checkpoint="dummy.pt",
        candidate_scaler="dummy.pkl"
    )


def test_promotion_gate_acceptance():
    """Verify promotion gate accepts high-performing models."""
    assert check_promotion_gate(
        candidate_model_name="test_transformer",
        candidate_rmse=170.0,
        candidate_r2=0.975,
        candidate_checkpoint="dummy.pt",
        candidate_scaler="dummy.pkl"
    )


def test_promotion_gate_mlflow_retrieval():
    """Verify promotion gate can extract metrics directly from logged MLflow runs."""
    passed = check_promotion_gate(
        candidate_model_name="transformer",
        candidate_checkpoint="model/transformer_model.pt",
        candidate_scaler="model/transformer_scaler.pkl"
    )
    assert passed is True
