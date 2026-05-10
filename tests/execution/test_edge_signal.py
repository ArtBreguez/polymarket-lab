"""Tests for EdgeSignal dataclass."""

from pmlab.execution.edge_signal import EdgeSignal


def test_creation_stores_fields():
    sig = EdgeSignal(
        market_id="mkt-001",
        city_or_segment="Chicago",
        target_date="2025-07-15",
        horizon="market_open",
        outcome_label="YES",
        direction="yes",
        gamma_price=0.6,
        model_prob=0.72,
        best_edge=0.12,
        yes_edge=0.12,
        no_edge=-0.02,
        question="Will it be hot?",
    )
    assert sig.market_id == "mkt-001"
    assert sig.city_or_segment == "Chicago"
    assert sig.target_date == "2025-07-15"
    assert sig.horizon == "market_open"
    assert sig.outcome_label == "YES"
    assert sig.direction == "yes"
    assert sig.gamma_price == 0.6
    assert sig.model_prob == 0.72
    assert sig.best_edge == 0.12
    assert sig.yes_edge == 0.12
    assert sig.no_edge == -0.02
    assert sig.question == "Will it be hot?"


def test_default_question_empty():
    sig = EdgeSignal(
        market_id="mkt-002",
        city_or_segment="Dallas",
        target_date="2025-08-01",
        horizon="morning_of",
        outcome_label="YES",
        direction="yes",
        gamma_price=0.55,
        model_prob=0.65,
        best_edge=0.10,
        yes_edge=0.10,
        no_edge=-0.05,
    )
    assert sig.question == ""
