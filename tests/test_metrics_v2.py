from __future__ import annotations

import math

import pytest

from utils.metrics_v2 import calculate_all_metrics


def test_metrics_v2_all_predictions_below_threshold_produce_no_trades():
    y_true = [0.0002, -0.0001, 0.0003]
    y_pred = [0.0001, -0.0001, 0.0002]

    metrics = calculate_all_metrics(
        y_true,
        y_pred,
        benchmark_values=None,
        risk_free=0.0,
        periods_per_year=6048,
        pip_size=0.0001,
        threshold_pips=5.0,
    )

    assert metrics["n_trades"] == 0
    assert metrics["hit_rate"] == 0.0
    assert math.isnan(metrics["avg_trade_return"])


def test_metrics_v2_hit_rate_uses_only_active_trades():
    y_true = [0.0010, -0.0010, 0.0020]
    y_pred = [0.0010, 0.0001, -0.0010]

    metrics = calculate_all_metrics(
        y_true,
        y_pred,
        benchmark_values=None,
        risk_free=0.0,
        periods_per_year=6048,
        pip_size=0.0001,
        threshold_pips=5.0,
    )

    assert metrics["n_trades"] == 2
    assert metrics["hit_rate"] == pytest.approx(50.0)


def test_metrics_v2_dm_stat_keeps_legacy_sign_when_model_is_better():
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]

    metrics = calculate_all_metrics(
        y_true,
        y_pred,
        benchmark_values=None,
        risk_free=0.0,
        periods_per_year=252,
        pip_size=0.0001,
        threshold_pips=0.0,
    )

    assert metrics["dm_stat"] < 0
    assert metrics["dm_pvalue"] <= 1.0


def test_metrics_v2_profitable_strategy_exposes_extended_metrics():
    y_true = [0.010, 0.012, -0.004, 0.008, -0.003, 0.009, -0.002, 0.006, 0.010, 0.011]
    y_pred = [0.011, 0.013, 0.003, 0.009, 0.004, 0.010, -0.003, 0.007, 0.011, 0.012]

    metrics = calculate_all_metrics(
        y_true,
        y_pred,
        benchmark_values=None,
        risk_free=0.0,
        periods_per_year=252,
        pip_size=0.0001,
        threshold_pips=5.0,
    )

    assert metrics["profit_factor"] > 1.0
    assert not math.isnan(metrics["calmar"])
    assert not math.isnan(metrics["consistency_ratio"])
    assert metrics["avg_trade_return"] > 0
