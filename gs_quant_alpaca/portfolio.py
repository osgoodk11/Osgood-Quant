"""
Paper-first Alpaca portfolio construction and rebalancing.
"""

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gs_quant.alpaca.client import AccountSnapshot, AlpacaBroker, PositionSnapshot, SubmittedOrder, build_alpaca_broker
from gs_quant.alpaca.config import AlpacaConfig


DEFAULT_UNIVERSE = ("SPY", "QQQ", "IWM", "TLT", "GLD")


@dataclass(frozen=True)
class PortfolioSettings:
    symbols: Tuple[str, ...] = DEFAULT_UNIVERSE
    history_days: int = 260
    fast_window: int = 20
    slow_window: int = 60
    momentum_window: int = 63
    max_positions: int = 5
    gross_target: float = 0.95
    max_symbol_weight: float = 0.25
    min_cash_pct: float = 0.02
    min_trade_notional: float = 25.0
    rebalance_tolerance_pct: float = 0.0025
    allow_fractional: bool = True

    def __post_init__(self):
        object.__setattr__(self, "symbols", _normalize_symbols(self.symbols))
        if not self.symbols:
            raise ValueError("At least one symbol is required.")
        if self.fast_window <= 0 or self.slow_window <= 0 or self.momentum_window <= 0:
            raise ValueError("Windows must be positive integers.")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive.")
        if not 0 < self.gross_target <= 1:
            raise ValueError("gross_target must be between 0 and 1.")
        if not 0 < self.max_symbol_weight <= 1:
            raise ValueError("max_symbol_weight must be between 0 and 1.")
        if not 0 <= self.min_cash_pct < 1:
            raise ValueError("min_cash_pct must be between 0 and 1.")


@dataclass(frozen=True)
class RebalanceOrder:
    symbol: str
    side: str
    quantity: float
    estimated_price: float
    estimated_notional: float
    current_weight: float
    target_weight: float


@dataclass(frozen=True)
class RebalancePlan:
    account: AccountSnapshot
    signals: pd.DataFrame
    target_weights: pd.Series
    latest_prices: pd.Series
    orders: Tuple[RebalanceOrder, ...]
    warnings: Tuple[str, ...] = ()
    submitted_orders: Tuple[SubmittedOrder, ...] = ()
    dry_run: bool = True

    @property
    def estimated_buy_notional(self) -> float:
        return sum(order.estimated_notional for order in self.orders if order.side == "buy")

    @property
    def estimated_sell_notional(self) -> float:
        return sum(order.estimated_notional for order in self.orders if order.side == "sell")


def compute_signal_frame(prices: pd.DataFrame, settings: PortfolioSettings) -> pd.DataFrame:
    """
    Compute simple trend and momentum signals from daily closes.

    Eligible symbols must have a positive momentum return, fast moving average
    above slow moving average, and latest close above the slow moving average.
    """

    rows = []
    for symbol in settings.symbols:
        series = prices.get(symbol, pd.Series(dtype=float)).dropna()
        minimum_points = max(settings.fast_window, settings.slow_window, settings.momentum_window) + 1
        if len(series) < minimum_points:
            rows.append(
                {
                    "symbol": symbol,
                    "latest_price": np.nan,
                    "fast_ma": np.nan,
                    "slow_ma": np.nan,
                    "momentum": np.nan,
                    "eligible": False,
                    "score": -np.inf,
                    "reason": "not_enough_history",
                }
            )
            continue

        fast_ma = _moving_average(series, settings.fast_window).dropna()
        slow_ma = _moving_average(series, settings.slow_window).dropna()
        momentum = _returns(series, settings.momentum_window).dropna()
        latest_price = float(series.iloc[-1])
        fast = float(fast_ma.iloc[-1]) if not fast_ma.empty else np.nan
        slow = float(slow_ma.iloc[-1]) if not slow_ma.empty else np.nan
        mom = float(momentum.iloc[-1]) if not momentum.empty else np.nan

        eligible = all(
            [
                np.isfinite(latest_price),
                np.isfinite(fast),
                np.isfinite(slow),
                np.isfinite(mom),
                latest_price > slow,
                fast > slow,
                mom > 0,
            ]
        )
        rows.append(
            {
                "symbol": symbol,
                "latest_price": latest_price,
                "fast_ma": fast,
                "slow_ma": slow,
                "momentum": mom,
                "eligible": bool(eligible),
                "score": mom if eligible else -np.inf,
                "reason": "eligible" if eligible else "trend_filter",
            }
        )

    signals = pd.DataFrame(rows).set_index("symbol")
    return signals.sort_values(["eligible", "score"], ascending=[False, False])


def target_weights_from_signals(signals: pd.DataFrame, settings: PortfolioSettings) -> pd.Series:
    target_weights = pd.Series(0.0, index=pd.Index(settings.symbols, name="symbol"))
    eligible = signals[signals["eligible"]].sort_values("score", ascending=False).head(settings.max_positions)
    if eligible.empty:
        return target_weights

    investable = min(settings.gross_target, 1.0 - settings.min_cash_pct)
    per_symbol = min(investable / len(eligible), settings.max_symbol_weight)
    for symbol in eligible.index:
        target_weights.loc[symbol] = per_symbol
    return target_weights


def build_rebalance_plan(
    account: AccountSnapshot,
    positions: Dict[str, PositionSnapshot],
    target_weights: pd.Series,
    latest_prices: pd.Series,
    settings: PortfolioSettings,
    signals: Optional[pd.DataFrame] = None,
) -> RebalancePlan:
    if account.equity <= 0:
        raise ValueError("Account equity must be positive.")

    orders = []
    warnings = []
    for symbol, target_weight in target_weights.items():
        position = positions.get(symbol)
        current_value = position.market_value if position else 0.0
        current_weight = current_value / account.equity
        target_value = account.equity * float(target_weight)
        delta = target_value - current_value
        price = _latest_price(symbol, latest_prices, position)

        if abs(delta) < settings.min_trade_notional:
            continue
        if abs(delta) / account.equity < settings.rebalance_tolerance_pct:
            continue
        if price is None or price <= 0:
            warnings.append(f"Skipped {symbol}: no usable latest price.")
            continue

        quantity = _quantity_from_delta(delta, price, settings.allow_fractional)
        if quantity == 0:
            continue

        side = "buy" if quantity > 0 else "sell"
        quantity_abs = abs(quantity)
        if side == "sell" and position and position.quantity > 0:
            quantity_abs = min(quantity_abs, position.quantity)

        if quantity_abs == 0:
            continue

        estimated_notional = quantity_abs * price
        if estimated_notional < settings.min_trade_notional:
            continue

        orders.append(
            RebalanceOrder(
                symbol=symbol,
                side=side,
                quantity=quantity_abs,
                estimated_price=price,
                estimated_notional=estimated_notional,
                current_weight=current_weight,
                target_weight=float(target_weight),
            )
        )

    sorted_orders = tuple(sorted(orders, key=lambda order: 0 if order.side == "sell" else 1))
    cash_after_estimate = account.cash + sum(
        order.estimated_notional if order.side == "sell" else -order.estimated_notional for order in sorted_orders
    )
    min_cash = account.equity * settings.min_cash_pct
    if cash_after_estimate < min_cash:
        warnings.append(
            f"Estimated cash after orders (${cash_after_estimate:,.2f}) is below the configured buffer "
            f"(${min_cash:,.2f})."
        )

    return RebalancePlan(
        account=account,
        signals=signals if signals is not None else pd.DataFrame(),
        target_weights=target_weights,
        latest_prices=latest_prices,
        orders=sorted_orders,
        warnings=tuple(warnings),
    )


def run_rebalance(
    broker: AlpacaBroker,
    settings: PortfolioSettings,
    dry_run: bool = True,
    today: Optional[dt.date] = None,
    config: Optional[AlpacaConfig] = None,
) -> RebalancePlan:
    today = today or dt.date.today()
    start = today - dt.timedelta(days=settings.history_days)
    end = today + dt.timedelta(days=1)

    prices = broker.get_price_history(settings.symbols, start, end)
    signals = compute_signal_frame(prices, settings)
    target_weights = target_weights_from_signals(signals, settings)
    latest_prices = prices.ffill().iloc[-1] if not prices.empty else pd.Series(dtype=float)
    account = broker.get_account()
    positions = broker.list_positions()
    plan = build_rebalance_plan(account, positions, target_weights, latest_prices, settings, signals)

    if dry_run:
        return plan

    if config is not None:
        config.assert_live_trading_allowed()
    if account.account_blocked or account.trading_blocked:
        raise RuntimeError("Alpaca account is blocked for trading.")

    submitted = []
    for order in plan.orders:
        submitted.append(broker.submit_market_order(order.symbol, order.side, order.quantity))
    return RebalancePlan(
        account=plan.account,
        signals=plan.signals,
        target_weights=plan.target_weights,
        latest_prices=plan.latest_prices,
        orders=plan.orders,
        warnings=plan.warnings,
        submitted_orders=tuple(submitted),
        dry_run=False,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build and rebalance a GS Quant Alpaca ETF portfolio.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE), help="Comma-separated tickers.")
    parser.add_argument("--execute", action="store_true", help="Submit the generated market orders.")
    parser.add_argument("--live", action="store_true", help="Use Alpaca live trading endpoints.")
    parser.add_argument("--feed", default=None, help="Alpaca market data feed, for example iex or sip.")
    parser.add_argument("--sdk", default="auto", choices=("auto", "alpaca-py", "alpaca-trade-api", "legacy"))
    parser.add_argument("--history-days", type=int, default=PortfolioSettings.history_days)
    parser.add_argument("--fast-window", type=int, default=PortfolioSettings.fast_window)
    parser.add_argument("--slow-window", type=int, default=PortfolioSettings.slow_window)
    parser.add_argument("--momentum-window", type=int, default=PortfolioSettings.momentum_window)
    parser.add_argument("--max-positions", type=int, default=PortfolioSettings.max_positions)
    parser.add_argument("--gross-target", type=float, default=PortfolioSettings.gross_target)
    parser.add_argument("--max-symbol-weight", type=float, default=PortfolioSettings.max_symbol_weight)
    parser.add_argument("--min-cash-pct", type=float, default=PortfolioSettings.min_cash_pct)
    parser.add_argument("--min-trade-notional", type=float, default=PortfolioSettings.min_trade_notional)
    parser.add_argument("--rebalance-tolerance-pct", type=float, default=PortfolioSettings.rebalance_tolerance_pct)
    parser.add_argument("--whole-shares", action="store_true", help="Round order quantities down to whole shares.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    args = parser.parse_args(argv)

    settings = PortfolioSettings(
        symbols=tuple(args.symbols.split(",")),
        history_days=args.history_days,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        momentum_window=args.momentum_window,
        max_positions=args.max_positions,
        gross_target=args.gross_target,
        max_symbol_weight=args.max_symbol_weight,
        min_cash_pct=args.min_cash_pct,
        min_trade_notional=args.min_trade_notional,
        rebalance_tolerance_pct=args.rebalance_tolerance_pct,
        allow_fractional=not args.whole_shares,
    )
    try:
        config = AlpacaConfig.from_env(paper=not args.live, data_feed=args.feed)
        broker = build_alpaca_broker(config, sdk=args.sdk)
        plan = run_rebalance(broker, settings, dry_run=not args.execute, config=config)
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_plan_to_json_dict(plan), indent=2, sort_keys=True))
    else:
        print(format_plan(plan))
    return 0


def format_plan(plan: RebalancePlan) -> str:
    lines = []
    mode = "DRY RUN" if plan.dry_run else "EXECUTED"
    lines.append(f"Alpaca portfolio rebalance ({mode})")
    lines.append(f"Equity: ${plan.account.equity:,.2f} | Cash: ${plan.account.cash:,.2f}")

    if not plan.signals.empty:
        signal_view = plan.signals[["latest_price", "fast_ma", "slow_ma", "momentum", "eligible"]].copy()
        lines.append("\nSignals")
        lines.append(signal_view.to_string(float_format=lambda value: f"{value:,.4f}"))

    weights = plan.target_weights[plan.target_weights > 0]
    lines.append("\nTarget weights")
    if weights.empty:
        lines.append("No symbols passed the trend filter; target is cash.")
    else:
        lines.append((weights * 100).round(2).astype(str).add("%").to_string())

    lines.append("\nOrders")
    if not plan.orders:
        lines.append("No orders needed.")
    else:
        order_frame = pd.DataFrame([asdict(order) for order in plan.orders])
        lines.append(order_frame.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    if plan.submitted_orders:
        lines.append("\nSubmitted")
        lines.append(pd.DataFrame([asdict(order) for order in plan.submitted_orders]).to_string(index=False))

    if plan.warnings:
        lines.append("\nWarnings")
        lines.extend(f"- {warning}" for warning in plan.warnings)

    return "\n".join(lines)


def _plan_to_json_dict(plan: RebalancePlan) -> dict:
    return {
        "dry_run": plan.dry_run,
        "account": asdict(plan.account),
        "target_weights": {symbol: float(value) for symbol, value in plan.target_weights.items()},
        "orders": [asdict(order) for order in plan.orders],
        "warnings": list(plan.warnings),
        "submitted_orders": [asdict(order) for order in plan.submitted_orders],
    }


def _normalize_symbols(symbols: Iterable[str]) -> Tuple[str, ...]:
    normalized = []
    seen = set()
    for symbol in symbols:
        clean = symbol.strip().upper()
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    return tuple(normalized)


def _latest_price(symbol: str, latest_prices: pd.Series, position: Optional[PositionSnapshot]) -> Optional[float]:
    value = latest_prices.get(symbol) if symbol in latest_prices.index else None
    if value is not None and np.isfinite(value) and value > 0:
        return float(value)
    if position and position.current_price > 0:
        return position.current_price
    return None


def _quantity_from_delta(delta: float, price: float, allow_fractional: bool) -> float:
    quantity = delta / price
    if allow_fractional:
        return round(quantity, 6)

    whole = math.floor(abs(quantity))
    return whole if quantity > 0 else -whole


def _moving_average(series: pd.Series, window: int) -> pd.Series:
    try:
        from gs_quant.timeseries.technicals import moving_average

        return moving_average(series, window)
    except ImportError:
        return series.rolling(window=window).mean()


def _returns(series: pd.Series, observations: int) -> pd.Series:
    try:
        from gs_quant.timeseries.econometrics import returns

        return returns(series, observations)
    except ImportError:
        return series / series.shift(observations) - 1
