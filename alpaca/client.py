"""
Broker adapters for Alpaca trading and stock bar data.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from gs_quant.alpaca.config import AlpacaConfig


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    trading_blocked: bool = False
    account_blocked: bool = False


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    quantity: float
    market_value: float
    current_price: float


@dataclass(frozen=True)
class SubmittedOrder:
    symbol: str
    side: str
    quantity: float
    order_id: Optional[str] = None
    status: Optional[str] = None


class AlpacaBroker:
    def get_price_history(self, symbols: Sequence[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError

    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError

    def list_positions(self) -> Dict[str, PositionSnapshot]:
        raise NotImplementedError

    def submit_market_order(self, symbol: str, side: str, quantity: float) -> SubmittedOrder:
        raise NotImplementedError


def build_alpaca_broker(config: AlpacaConfig, sdk: str = "auto") -> AlpacaBroker:
    """
    Build an Alpaca broker adapter.

    The official alpaca-py SDK is preferred when installed. The older
    alpaca-trade-api package is still supported because many local quant
    environments already have it.
    """

    sdk = sdk.lower()
    errors = []
    if sdk in {"auto", "alpaca-py", "alpaca_py"}:
        try:
            return AlpacaPyBroker(config)
        except ImportError as exc:
            errors.append(str(exc))

    if sdk in {"auto", "alpaca-trade-api", "legacy"}:
        try:
            return LegacyAlpacaBroker(config)
        except ImportError as exc:
            errors.append(str(exc))

    raise ImportError(
        "Install alpaca-py or alpaca-trade-api to use gs_quant.alpaca. "
        f"Tried SDK mode '{sdk}'. Details: {'; '.join(errors)}"
    )


class LegacyAlpacaBroker(AlpacaBroker):
    def __init__(self, config: AlpacaConfig, rest_client=None):
        if rest_client is None:
            try:
                from alpaca_trade_api import REST
            except ImportError as exc:
                raise ImportError("alpaca-trade-api is not installed") from exc
            rest_client = REST(
                key_id=config.api_key,
                secret_key=config.secret_key,
                base_url=config.base_url,
                api_version="v2",
            )
        self.config = config
        self._rest = rest_client

    def get_price_history(self, symbols: Sequence[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        from alpaca_trade_api.rest import TimeFrame

        kwargs = {}
        if self.config.data_feed:
            feed = self.config.data_feed
            try:
                from alpaca.data.enums import DataFeed

                feed = DataFeed(feed)
            except (ImportError, ValueError):
                pass
            kwargs["feed"] = feed
        bars = self._rest.get_bars(
            list(symbols),
            TimeFrame.Day,
            start=start.isoformat(),
            end=end.isoformat(),
            adjustment="raw",
            **kwargs,
        )
        return _bars_to_close_frame(getattr(bars, "df", bars), symbols)

    def get_account(self) -> AccountSnapshot:
        account = self._rest.get_account()
        return AccountSnapshot(
            equity=_float_attr(account, "equity"),
            cash=_float_attr(account, "cash"),
            buying_power=_float_attr(account, "buying_power"),
            trading_blocked=bool(_attr(account, "trading_blocked", default=False)),
            account_blocked=bool(_attr(account, "account_blocked", default=False)),
        )

    def list_positions(self) -> Dict[str, PositionSnapshot]:
        positions = {}
        for position in self._rest.list_positions():
            snapshot = _position_snapshot(position)
            positions[snapshot.symbol] = snapshot
        return positions

    def submit_market_order(self, symbol: str, side: str, quantity: float) -> SubmittedOrder:
        order = self._rest.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type="market",
            time_in_force="day",
        )
        return SubmittedOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_id=str(_attr(order, "id", default="")) or None,
            status=_attr(order, "status"),
        )


class AlpacaPyBroker(AlpacaBroker):
    def __init__(self, config: AlpacaConfig):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError("alpaca-py is not installed") from exc

        self.config = config
        try:
            self._trading = TradingClient(
                api_key=config.api_key,
                secret_key=config.secret_key,
                paper=config.paper,
                url_override=config.base_url,
            )
        except TypeError:
            self._trading = TradingClient(api_key=config.api_key, secret_key=config.secret_key, paper=config.paper)
        self._data = StockHistoricalDataClient(config.api_key, config.secret_key)

    def get_price_history(self, symbols: Sequence[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        kwargs = {}
        if self.config.data_feed:
            kwargs["feed"] = self.config.data_feed
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Day,
            start=dt.datetime.combine(start, dt.time.min),
            end=dt.datetime.combine(end, dt.time.min),
            **kwargs,
        )
        bars = self._data.get_stock_bars(request)
        return _bars_to_close_frame(getattr(bars, "df", bars), symbols)

    def get_account(self) -> AccountSnapshot:
        account = self._trading.get_account()
        return AccountSnapshot(
            equity=_float_attr(account, "equity"),
            cash=_float_attr(account, "cash"),
            buying_power=_float_attr(account, "buying_power"),
            trading_blocked=bool(_attr(account, "trading_blocked", default=False)),
            account_blocked=bool(_attr(account, "account_blocked", default=False)),
        )

    def list_positions(self) -> Dict[str, PositionSnapshot]:
        positions = {}
        for position in self._trading.get_all_positions():
            snapshot = _position_snapshot(position)
            positions[snapshot.symbol] = snapshot
        return positions

    def submit_market_order(self, symbol: str, side: str, quantity: float) -> SubmittedOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._trading.submit_order(order_data=request)
        return SubmittedOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_id=str(_attr(order, "id", default="")) or None,
            status=_attr(order, "status"),
        )


def _attr(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _float_attr(obj, name: str, default: float = 0.0) -> float:
    value = _attr(obj, name, default)
    if value is None:
        return default
    return float(value)


def _position_snapshot(position) -> PositionSnapshot:
    symbol = str(_attr(position, "symbol")).upper()
    quantity = _float_attr(position, "qty")
    market_value = _float_attr(position, "market_value")
    current_price = _float_attr(position, "current_price")
    side = str(_attr(position, "side", default="long")).lower()
    if side == "short":
        quantity = -abs(quantity)
        market_value = -abs(market_value)
    return PositionSnapshot(
        symbol=symbol,
        quantity=quantity,
        market_value=market_value,
        current_price=current_price,
    )


def _bars_to_close_frame(raw_bars, symbols: Sequence[str]) -> pd.DataFrame:
    if raw_bars is None:
        return pd.DataFrame(columns=list(symbols))

    df = pd.DataFrame(raw_bars).copy()
    if df.empty:
        return pd.DataFrame(columns=list(symbols))

    df = df.reset_index()
    close_col = "close"
    if close_col not in df.columns:
        raise ValueError("Alpaca bars did not include a close column.")

    symbol_col = _find_symbol_column(df, symbols)
    if symbol_col is None:
        if len(symbols) == 1:
            symbol_col = "symbol"
            df[symbol_col] = symbols[0]
        else:
            raise ValueError("Alpaca bars did not include symbol information for a multi-symbol request.")

    time_col = _find_time_column(df)
    if time_col is None:
        raise ValueError("Alpaca bars did not include a timestamp column.")

    timestamps = pd.to_datetime(df[time_col])
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert(None)
    df["_date"] = timestamps.dt.normalize()
    df[symbol_col] = df[symbol_col].astype(str).str.upper()

    closes = df.pivot_table(index="_date", columns=symbol_col, values=close_col, aggfunc="last")
    ordered = [symbol for symbol in symbols if symbol in closes.columns]
    return closes.reindex(columns=ordered).sort_index()


def _find_symbol_column(df: pd.DataFrame, symbols: Sequence[str]) -> Optional[str]:
    if "symbol" in df.columns:
        return "symbol"

    expected = {symbol.upper() for symbol in symbols}
    for column in df.columns:
        values = df[column].dropna()
        if values.empty or pd.api.types.is_datetime64_any_dtype(values):
            continue
        as_strings = {str(value).upper() for value in values.unique()}
        if as_strings and as_strings.issubset(expected):
            return column
    return None


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    for column in ("timestamp", "time", "date", "index", "level_0", "level_1"):
        if column in df.columns and _is_datetime_like(df[column]):
            return column

    for column in df.columns:
        if _is_datetime_like(df[column]):
            return column
    return None


def _is_datetime_like(values: Iterable) -> bool:
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return False
    try:
        converted = pd.to_datetime(series, errors="coerce")
    except (TypeError, ValueError):
        return False
    return not converted.isna().all()
