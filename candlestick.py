"""
Pocket AI Trader — Candlestick Engine (Phase 6)
==================================================

Analyse la dernière bougie close (et son contexte proche) pour en extraire :
- ses caractéristiques géométriques (corps, mèches, ratio) ;
- les patterns de chandeliers japonais reconnus (§8 de la spec).

Rappel du principe : un pattern isolé n'a que peu de valeur — les agents IA
(Phase 8) combineront ces patterns avec les indicateurs (Phase 5) et le
contexte de tendance. Ce module se contente de la détection brute.
"""

from dataclasses import dataclass, field
from typing import Optional

from market_data import Candle


@dataclass
class CandleGeometry:
    body: float
    upper_wick: float
    lower_wick: float
    range_: float
    body_wick_ratio: Optional[float]  # corps / (mèche haute + mèche basse)
    is_bullish: bool


def _geometry(c: Candle) -> CandleGeometry:
    body = abs(c.close - c.open)
    upper_wick = c.high - max(c.close, c.open)
    lower_wick = min(c.close, c.open) - c.low
    range_ = c.high - c.low
    total_wick = upper_wick + lower_wick
    ratio = body / total_wick if total_wick > 0 else None
    return CandleGeometry(
        body=body,
        upper_wick=upper_wick,
        lower_wick=lower_wick,
        range_=range_,
        body_wick_ratio=ratio,
        is_bullish=c.close > c.open,
    )


@dataclass
class CandleAnalysis:
    instrument: str
    timeframe: str
    geometry: CandleGeometry
    patterns: list[str] = field(default_factory=list)
    context: str = ""  # "uptrend" | "downtrend" | "range"

    def summary_lines(self) -> list[str]:
        g = self.geometry
        ratio = f"{g.body_wick_ratio:.2f}" if g.body_wick_ratio is not None else "—"
        lines = [
            f"Sens : {'Haussière 🟢' if g.is_bullish else 'Baissière 🔴'}",
            f"Contexte récent : {self.context}",
            f"Corps/mèches : {ratio}",
        ]
        if self.patterns:
            lines.append("Patterns détectés :")
            lines += [f"✓ {p}" for p in self.patterns]
        else:
            lines.append("Aucun pattern net détecté sur cette bougie.")
        return lines


def _recent_trend(candles: list[Candle], lookback: int = 10) -> str:
    """Détermine grossièrement si les bougies précédentes forment une
    tendance haussière, baissière, ou un range — sert de contexte pour
    les patterns de retournement (Hammer, Shooting Star, etc.)."""
    window = candles[-lookback - 1 : -1]  # exclut la bougie courante
    if len(window) < 3:
        return "range"
    closes = [c.close for c in window]
    change = closes[-1] - closes[0]
    avg_range = sum(c.high - c.low for c in window) / len(window)
    if avg_range == 0:
        return "range"
    if change > avg_range * 1.5:
        return "uptrend"
    if change < -avg_range * 1.5:
        return "downtrend"
    return "range"


def _detect_patterns(candles: list[Candle]) -> list[str]:
    """candles : liste chronologique, la dernière étant la bougie à analyser.
    Utilise jusqu'à 3 bougies de contexte pour les patterns multi-bougies."""
    patterns: list[str] = []
    if len(candles) < 1:
        return patterns

    c0 = candles[-1]  # bougie analysée
    g0 = _geometry(c0)
    trend = _recent_trend(candles)

    # --- Patterns à 1 bougie ---
    if g0.range_ > 0 and g0.body <= 0.1 * g0.range_:
        patterns.append("Doji")

    if g0.body > 0 and g0.range_ > 0 and g0.body >= 0.9 * g0.range_:
        patterns.append("Marubozu")

    # Hammer : petit corps en haut du range, longue mèche basse, après une baisse
    if (
        g0.range_ > 0
        and g0.lower_wick >= 2 * g0.body
        and g0.upper_wick <= 0.1 * g0.range_
        and trend == "downtrend"
    ):
        patterns.append("Hammer (retournement haussier potentiel)")

    # Shooting Star : petit corps en bas du range, longue mèche haute, après une hausse
    if (
        g0.range_ > 0
        and g0.upper_wick >= 2 * g0.body
        and g0.lower_wick <= 0.1 * g0.range_
        and trend == "uptrend"
    ):
        patterns.append("Shooting Star (retournement baissier potentiel)")

    # Pin Bar générique (une seule mèche dominante, corps petit, sans exigence de tendance)
    if g0.range_ > 0 and g0.body <= 0.3 * g0.range_:
        if g0.lower_wick >= 2 * g0.upper_wick and g0.lower_wick >= 0.5 * g0.range_:
            patterns.append("Pin Bar haussier")
        elif g0.upper_wick >= 2 * g0.lower_wick and g0.upper_wick >= 0.5 * g0.range_:
            patterns.append("Pin Bar baissier")

    # --- Patterns à 2 bougies ---
    if len(candles) >= 2:
        c1 = candles[-2]
        g1 = _geometry(c1)

        # Engulfing : le corps de c0 englobe entièrement celui de c1, couleur opposée
        c0_top, c0_bot = max(c0.open, c0.close), min(c0.open, c0.close)
        c1_top, c1_bot = max(c1.open, c1.close), min(c1.open, c1.close)
        if g0.is_bullish != g1.is_bullish and c0_top >= c1_top and c0_bot <= c1_bot and g0.body > 0:
            patterns.append(
                "Bullish Engulfing" if g0.is_bullish else "Bearish Engulfing"
            )

        # Harami : corps de c0 contenu dans celui de c1 (inverse de l'engulfing)
        if c0_top <= c1_top and c0_bot >= c1_bot and g1.body > 0 and g0.body < g1.body:
            patterns.append("Harami")

        # Inside Bar : toute la bougie c0 (high/low) contenue dans c1
        if c0.high <= c1.high and c0.low >= c1.low:
            patterns.append("Inside Bar")

        # Tweezer Top / Bottom : sommets ou creux quasi identiques
        tolerance = (g1.range_ + g0.range_) / 2 * 0.1 if (g1.range_ + g0.range_) > 0 else 0
        if abs(c0.high - c1.high) <= tolerance and trend == "uptrend":
            patterns.append("Tweezer Top")
        if abs(c0.low - c1.low) <= tolerance and trend == "downtrend":
            patterns.append("Tweezer Bottom")

        # Breakout Candle : corps nettement plus grand que la moyenne récente + clôture au-delà du plus haut/bas récent
        recent = candles[-6:-1] if len(candles) >= 6 else candles[:-1]
        if recent:
            avg_body = sum(_geometry(c).body for c in recent) / len(recent)
            recent_high = max(c.high for c in recent)
            recent_low = min(c.low for c in recent)
            if avg_body > 0 and g0.body >= 1.8 * avg_body:
                if c0.close > recent_high:
                    patterns.append("Breakout Candle (haussier)")
                elif c0.close < recent_low:
                    patterns.append("Breakout Candle (baissier)")

        # Rejection Candle : longue mèche qui rejette un niveau récent (plus haut/bas des bougies précédentes)
        if recent:
            recent_high = max(c.high for c in recent)
            recent_low = min(c.low for c in recent)
            if g0.upper_wick >= 0.6 * g0.range_ and c0.high >= recent_high and g0.range_ > 0:
                patterns.append("Rejection Candle (résistance)")
            if g0.lower_wick >= 0.6 * g0.range_ and c0.low <= recent_low and g0.range_ > 0:
                patterns.append("Rejection Candle (support)")

    # --- Patterns à 3 bougies ---
    if len(candles) >= 3:
        c2, c1 = candles[-3], candles[-2]
        g2, g1 = _geometry(c2), _geometry(c1)

        # Morning Star : bougie baissière, petite bougie (indécision), bougie haussière qui referme > moitié de c2
        if (
            not g2.is_bullish
            and g1.body <= 0.4 * g2.body
            and g0.is_bullish
            and c0.close > (c2.open + c2.close) / 2
        ):
            patterns.append("Morning Star")

        # Evening Star : inverse
        if (
            g2.is_bullish
            and g1.body <= 0.4 * g2.body
            and not g0.is_bullish
            and c0.close < (c2.open + c2.close) / 2
        ):
            patterns.append("Evening Star")

    return patterns


def analyze_candles(instrument: str, timeframe: str, candles: list[Candle]) -> CandleAnalysis:
    """candles : liste chronologique d'au moins quelques bougies (idéalement
    10+ pour un contexte de tendance fiable), la dernière étant analysée."""
    geometry = _geometry(candles[-1])
    patterns = _detect_patterns(candles)
    context = _recent_trend(candles)
    return CandleAnalysis(
        instrument=instrument,
        timeframe=timeframe,
        geometry=geometry,
        patterns=patterns,
        context=context,
    )
