"""Unit tests for MLOps drift detection, PSI calculation, and model registry lifecycle."""
import numpy as np
import pytest
from src.mlops.drift_monitor import DriftMonitor, DriftSeverity, PSICalculator
from src.mlops.model_registry import ModelRegistry, ModelStage
from src.mlops.synthetic_data import SyntheticFraudGenerator


@pytest.mark.unit
def test_psi_calculator_stability_and_drift():
    """PSICalculator should accurately detect Stable, Marginal Shift, and Severe Drift."""
    np.random.seed(42)
    # 1. Identical distributions -> PSI < 0.10 (Stable)
    baseline = np.random.normal(loc=100.0, scale=10.0, size=500)
    actual_stable = np.random.normal(loc=100.0, scale=10.0, size=500)
    psi_stable = PSICalculator.calculate_psi(baseline, actual_stable)
    assert psi_stable < 0.10

    # 2. Moderately shifted distribution -> Marginal shift
    actual_marginal = np.random.normal(loc=106.0, scale=12.0, size=500)
    psi_marginal = PSICalculator.calculate_psi(baseline, actual_marginal)
    assert psi_marginal > psi_stable

    # 3. Severely shifted distribution -> Severe drift (PSI >= 0.25)
    actual_severe = np.random.normal(loc=150.0, scale=30.0, size=500)
    psi_severe = PSICalculator.calculate_psi(baseline, actual_severe)
    assert psi_severe >= 0.25


@pytest.mark.unit
def test_drift_monitor_alerting():
    """DriftMonitor should trigger alerts and flags when drift threshold is crossed."""
    monitor = DriftMonitor()
    baseline = list(np.random.normal(50.0, 5.0, 100))
    monitor.set_baseline("invoice_amount_vs_90d_avg_ratio", baseline)

    # Feed severely drifted production values
    for val in np.random.normal(120.0, 20.0, 100):
        monitor.record_production_value("invoice_amount_vs_90d_avg_ratio", val)

    report = monitor.evaluate_all()
    assert report.overall_severity in (DriftSeverity.MARGINAL_SHIFT, DriftSeverity.SEVERE_DRIFT)
    assert report.requires_shadow_retraining is True


@pytest.mark.unit
def test_model_registry_promotion(tmp_path):
    """ModelRegistry should track versions, promote stages, and archive superseded models."""
    reg = ModelRegistry(registry_dir=str(tmp_path / "models"))

    m1 = reg.register_model(
        model_name="p2p-fraud-lightgbm",
        version="1.0.0",
        algorithm="LightGBM",
        auc_pr=0.885,
        brier_score=0.042,
        training_sample_count=10000,
        artifact_path="/tmp/m1.joblib",
        stage=ModelStage.SHADOW,
    )
    assert m1.stage == ModelStage.SHADOW

    # Promote to production
    reg.promote_model("p2p-fraud-lightgbm", "1.0.0", ModelStage.PRODUCTION)
    prod = reg.get_production_model("p2p-fraud-lightgbm")
    assert prod is not None
    assert prod.version == "1.0.0"

    # Register and promote v2.0.0
    reg.register_model(
        model_name="p2p-fraud-lightgbm",
        version="2.0.0",
        algorithm="LightGBM",
        auc_pr=0.912,
        brier_score=0.035,
        training_sample_count=20000,
        artifact_path="/tmp/m2.joblib",
        stage=ModelStage.SHADOW,
    )
    reg.promote_model("p2p-fraud-lightgbm", "2.0.0", ModelStage.PRODUCTION)

    # v2.0.0 is now production, v1.0.0 is archived
    new_prod = reg.get_production_model("p2p-fraud-lightgbm")
    assert new_prod.version == "2.0.0"
    assert reg.models["p2p-fraud-lightgbm:1.0.0"].stage == ModelStage.ARCHIVED


@pytest.mark.unit
def test_synthetic_fraud_generator():
    """Synthetic generator should produce valid structured vectors."""
    clean_batch = SyntheticFraudGenerator.generate_clean_batch(n_samples=50)
    assert clean_batch.shape == (50, 14)

    bank_scen = SyntheticFraudGenerator.generate_fraud_scenario("MID_CYCLE_BANK_DIVERSION")
    assert bank_scen["is_new_bank_account_72h"] == 1.0
    assert bank_scen["vendor_bank_account_age_days"] == 1.5

    smurf_scen = SyntheticFraudGenerator.generate_fraud_scenario("SMURFING_SPLIT_INVOICE")
    assert smurf_scen["smurfing_24h_pattern_detected"] == 1.0
