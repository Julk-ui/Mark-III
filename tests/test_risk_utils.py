import pytest

from utils.risk_utils import (
    calculate_position_size_for_risk_amount,
    estimate_position_risk_amount,
)


def test_estimate_position_risk_amount_for_one_lot_and_ten_pips():
    risk_amount = estimate_position_risk_amount(
        entry_price=1.1000,
        sl_price=1.0990,
        point=0.0001,
        contract_size=100000.0,
        volume_lots=1.0,
    )
    assert risk_amount == pytest.approx(100.0)


def test_calculate_position_size_for_fixed_risk_amount():
    lots = calculate_position_size_for_risk_amount(
        entry_price=1.1000,
        sl_price=1.0990,
        point=0.0001,
        contract_size=100000.0,
        risk_amount=100.0,
        min_lot=0.01,
        lot_step=0.01,
    )
    assert lots == 1.0


def test_calculate_position_size_returns_zero_when_budget_is_below_min_lot():
    lots = calculate_position_size_for_risk_amount(
        entry_price=1.1000,
        sl_price=1.0990,
        point=0.0001,
        contract_size=100000.0,
        risk_amount=0.5,
        min_lot=0.01,
        lot_step=0.01,
    )
    assert lots == 0.0
