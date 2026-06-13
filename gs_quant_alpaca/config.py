"""
Configuration helpers for Alpaca paper and live accounts.
"""

import os
from dataclasses import dataclass
from typing import Optional


PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AlpacaConfig:
    """
    Alpaca account and endpoint settings.

    Environment variable names follow Alpaca's common APCA naming while also
    accepting clearer ALPACA aliases:

    * ALPACA_API_KEY_ID or APCA_API_KEY_ID
    * ALPACA_SECRET_KEY or APCA_API_SECRET_KEY
    * ALPACA_BASE_URL or APCA_API_BASE_URL
    * ALPACA_DATA_FEED, commonly "iex" or "sip"
    """

    api_key: str
    secret_key: str
    base_url: str = PAPER_BASE_URL
    paper: bool = True
    data_feed: Optional[str] = None

    @classmethod
    def from_env(cls, paper: Optional[bool] = None, data_feed: Optional[str] = None) -> "AlpacaConfig":
        api_key = _first_env("ALPACA_API_KEY_ID", "APCA_API_KEY_ID", "APCA_API_KEY")
        secret_key = _first_env("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY", "APCA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "Set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY, or Alpaca's APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY variables."
            )

        if paper is None:
            paper = _env_bool("ALPACA_PAPER", True)

        default_base_url = PAPER_BASE_URL if paper else LIVE_BASE_URL
        base_url = _first_env("ALPACA_BASE_URL", "APCA_API_BASE_URL") or default_base_url
        resolved_feed = data_feed or _first_env("ALPACA_DATA_FEED", "APCA_DATA_FEED")

        return cls(api_key=api_key, secret_key=secret_key, base_url=base_url, paper=paper, data_feed=resolved_feed)

    @property
    def is_live(self) -> bool:
        return not self.paper or "paper-api" not in self.base_url.lower()

    def assert_live_trading_allowed(self) -> None:
        if self.is_live and not _env_bool("ALPACA_ALLOW_LIVE_TRADING", False):
            raise RuntimeError(
                "Live Alpaca trading is disabled. Set ALPACA_ALLOW_LIVE_TRADING=1 only after you have reviewed "
                "the generated order plan."
            )
