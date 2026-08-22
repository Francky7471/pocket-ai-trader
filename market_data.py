"""
Pocket AI Trader — Market Data Engine (Phase 4)
=================================================

Récupère les bougies (OHLC) en direct pour les instruments Forex suivis,
sur les 3 timeframes définis en Phase 1 (H1 / M15 / M5), via l'API
Twelve Data (offre gratuite).

Ce module ne fait QUE de la récupération de données — les indicateurs
(Phase 5) et l'analyse des chandeliers (Phase 6) travailleront à partir
de ce qu'il renvoie.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp

logger = logging.getLogger("pocket_ai_trader.market_data")

API_KEY = os.environ["TWELVE_DATA_API_KEY"]
BASE_URL = "https://api.twelvedata.com/time_series"

# Twelve Data utilise le format "EUR/USD" ; nos instruments internes sont
# notés "EURUSD" (comme MT5) — conversion faite ici pour rester cohérent
# avec le reste du système.
def _to_twelvedata_symbol(instrument: str) -> str:
    instrument = instrument.upper().replace("/", "")
    return f"{instrument[:3]}/{instrument[3:]}"


# Nos timeframes internes -> format attendu par Twelve Data
_INTERVAL_MAP = {
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
}


@dataclass
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float


class MarketDataError(Exception):
    pass


async def get_candles(instrument: str, timeframe: str, count: int = 50) -> list[Candle]:
    """Renvoie les `count` dernières bougies pour un instrument/timeframe,
    triées de la plus ancienne à la plus récente."""
    if timeframe not in _INTERVAL_MAP:
        raise ValueError(f"Timeframe non supporté : {timeframe}")

    params = {
        "symbol": _to_twelvedata_symbol(instrument),
        "interval": _INTERVAL_MAP[timeframe],
        "outputsize": str(count),
        "apikey": API_KEY,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            data = await resp.json()

    if data.get("status") == "error":
        raise MarketDataError(data.get("message", "Erreur inconnue de Twelve Data."))

    values = data.get("values")
    if not values:
        raise MarketDataError(f"Aucune donnée reçue pour {instrument} {timeframe}.")

    candles = [
        Candle(
            time=datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S")
            if len(v["datetime"]) > 10
            else datetime.strptime(v["datetime"], "%Y-%m-%d"),
            open=float(v["open"]),
            high=float(v["high"]),
            low=float(v["low"]),
            close=float(v["close"]),
        )
        for v in values
    ]
    candles.reverse()  # Twelve Data renvoie du plus récent au plus ancien
    return candles


async def get_latest_price(instrument: str) -> Optional[float]:
    """Dernier prix de clôture connu (bougie M5), pratique pour un test rapide."""
    try:
        candles = await get_candles(instrument, "M5", count=1)
        return candles[-1].close if candles else None
    except MarketDataError as e:
        logger.warning(f"Impossible de récupérer le prix de {instrument} : {e}")
        return None
