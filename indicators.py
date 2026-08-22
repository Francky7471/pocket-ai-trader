"""
Pocket AI Trader — Indicator Engine (Phase 5)
================================================

Transforme les bougies brutes (fournies par market_data.py) en indicateurs
techniques exploitables. Utilise la librairie `ta` (pure Python, pas de
dépendance C à compiler — contrairement à TA-Lib, plus fragile à déployer
sur un hébergement gratuit comme Render).

Ce module renvoie un "instantané" des indicateurs à l'instant présent
(dernière bougie close) — c'est ce que les 5 agents IA consommeront à
partir de la Phase 8.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from market_data import Candle, get_candles, MarketDataError

logger = logging.getLogger("pocket_ai_trader.indicators")


@dataclass
class IndicatorSnapshot:
    instrument: str
    timeframe: str
    close: float
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]
    ema9: Optional[float]
    ema21: Optional[float]
    ema50: Optional[float]
    ema200: Optional[float]
    sma20: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    atr: Optional[float]
    adx: Optional[float]
    stoch_k: Optional[float]
    stoch_d: Optional[float]
    cci: Optional[float]
    williams_r: Optional[float]
    roc: Optional[float]

    def summary_lines(self) -> list[str]:
        """Format lisible pour un affichage Telegram."""
        def fmt(v, decimals=2):
            return f"{v:.{decimals}f}" if v is not None else "—"

        return [
            f"Clôture : {self.close}",
            f"RSI(14) : {fmt(self.rsi)}",
            f"MACD : {fmt(self.macd, 5)} / signal {fmt(self.macd_signal, 5)}",
            f"EMA 9/21/50/200 : {fmt(self.ema9, 5)} / {fmt(self.ema21, 5)} / {fmt(self.ema50, 5)} / {fmt(self.ema200, 5)}",
            f"Bollinger (haut/bas) : {fmt(self.bb_upper, 5)} / {fmt(self.bb_lower, 5)}",
            f"ATR(14) : {fmt(self.atr, 5)}",
            f"ADX(14) : {fmt(self.adx)}",
            f"Stochastic %K/%D : {fmt(self.stoch_k)} / {fmt(self.stoch_d)}",
            f"CCI(20) : {fmt(self.cci)}",
            f"Williams %R : {fmt(self.williams_r)}",
            f"ROC(12) : {fmt(self.roc)}",
        ]


def _candles_to_df(candles: list[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
        }
    )


def compute_indicators(instrument: str, timeframe: str, candles: list[Candle]) -> IndicatorSnapshot:
    df = _candles_to_df(candles)
    close, high, low = df["close"], df["high"], df["low"]

    def last(series) -> Optional[float]:
        try:
            val = series.iloc[-1]
            return float(val) if pd.notna(val) else None
        except Exception:
            return None

    rsi = RSIIndicator(close, window=14).rsi()
    macd_ind = MACD(close)
    ema9 = EMAIndicator(close, window=9).ema_indicator()
    ema21 = EMAIndicator(close, window=21).ema_indicator()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    sma20 = SMAIndicator(close, window=20).sma_indicator()
    bb = BollingerBands(close, window=20)
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    adx = ADXIndicator(high, low, close, window=14).adx()
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    cci = CCIIndicator(high, low, close, window=20).cci()
    williams = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    roc = ROCIndicator(close, window=12).roc()

    return IndicatorSnapshot(
        instrument=instrument,
        timeframe=timeframe,
        close=float(close.iloc[-1]),
        rsi=last(rsi),
        macd=last(macd_ind.macd()),
        macd_signal=last(macd_ind.macd_signal()),
        macd_hist=last(macd_ind.macd_diff()),
        ema9=last(ema9),
        ema21=last(ema21),
        ema50=last(ema50),
        ema200=last(ema200),
        sma20=last(sma20),
        bb_upper=last(bb.bollinger_hband()),
        bb_lower=last(bb.bollinger_lband()),
        atr=last(atr),
        adx=last(adx),
        stoch_k=last(stoch.stoch()),
        stoch_d=last(stoch.stoch_signal()),
        cci=last(cci),
        williams_r=last(williams),
        roc=last(roc),
    )


async def analyze_instrument(instrument: str, timeframe: str = "M5", count: int = 210) -> Optional[IndicatorSnapshot]:
    """Récupère les bougies et calcule tous les indicateurs pour un
    instrument/timeframe donné. Renvoie None si les données sont
    indisponibles (ex. instrument invalide, quota API dépassé)."""
    try:
        candles = await get_candles(instrument, timeframe, count=count)
    except MarketDataError as e:
        logger.warning(f"Impossible d'analyser {instrument} {timeframe} : {e}")
        return None

    if len(candles) < 20:
        logger.warning(f"Pas assez de bougies pour {instrument} {timeframe} ({len(candles)}).")
        return None

    return compute_indicators(instrument, timeframe, candles)
