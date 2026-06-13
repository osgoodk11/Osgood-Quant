"""
Lightweight Alpaca portfolio tooling for GS Quant analytics.

The public surface here is intentionally small: build a broker client from
environment variables, compute target weights from Alpaca price history, and
create or execute a rebalance plan.
"""

from gs_quant.alpaca.client import AccountSnapshot, AlpacaBroker, PositionSnapshot, build_alpaca_broker
from gs_quant.alpaca.config import AlpacaConfig
from gs_quant.alpaca.portfolio import (
    PortfolioSettings,
    RebalanceOrder,
    RebalancePlan,
    build_rebalance_plan,
    compute_signal_frame,
    target_weights_from_signals,
)

__all__ = [
    "AccountSnapshot",
    "AlpacaBroker",
    "AlpacaConfig",
    "PortfolioSettings",
    "PositionSnapshot",
    "RebalanceOrder",
    "RebalancePlan",
    "build_alpaca_broker",
    "build_rebalance_plan",
    "compute_signal_frame",
    "target_weights_from_signals",
]
