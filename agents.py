"""
Pocket AI Trader — 5 Agents IA indépendants (Phase 8)
========================================================

Version "règles expertes" (choisie pour rester 100% gratuite — pas d'appel
à une API IA payante). Chaque agent applique sa propre méthodologie, comme
prévu dans la spec initiale (§4), en s'appuyant sur les briques déjà
construites (indicateurs, chandeliers) plutôt qu'un raisonnement en langage
naturel. Chaque agent répond indépendamment : CALL / PUT / NO_TRADE +
score de confiance + raisons — exactement le même format qu'un agent LLM
aurait renvoyé, pour que les Validateurs (Phase 9) n'aient pas à faire de
distinction.
"""

from dataclasses import dataclass, field
from typing import Optional

from indicators import IndicatorSnapshot
from candlestick import CandleAnalysis
from market_data import Candle
from strategy import Direction, REVERSAL_BULLISH_PATTERNS, REVERSAL_BEARISH_PATTERNS


@dataclass
class AgentSignal:
    agent_name: str
    direction: Direction
    confidence: int  # 0-100
    reasons: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        icon = {"CALL": "🟢", "PUT": "🔴", "NO_TRADE": "⚪"}[self.direction.value]
        return f"{icon} {self.agent_name} : {self.direction.value} ({self.confidence}/100)"


def _no_trade(name: str, reason: str) -> AgentSignal:
    return AgentSignal(name, Direction.NO_TRADE, 0, [reason])


# --- Agent 1 : Momentum ----------------------------------------------------
def agent_momentum(snap: IndicatorSnapshot, candle: CandleAnalysis) -> AgentSignal:
    name = "Agent 1 — Momentum"
    if None in (snap.rsi, snap.macd, snap.macd_signal, snap.stoch_k, snap.stoch_d, snap.roc):
        return _no_trade(name, "Données insuffisantes.")

    bullish_votes = sum([
        snap.rsi > 50,
        snap.macd > snap.macd_signal,
        snap.stoch_k > snap.stoch_d and snap.stoch_k < 80,
        snap.roc > 0,
    ])
    bearish_votes = sum([
        snap.rsi < 50,
        snap.macd < snap.macd_signal,
        snap.stoch_k < snap.stoch_d and snap.stoch_k > 20,
        snap.roc < 0,
    ])

    if bullish_votes >= 3:
        confidence = 55 + bullish_votes * 8
        return AgentSignal(name, Direction.CALL, min(confidence, 95), [
            f"RSI = {snap.rsi:.1f}", f"MACD > signal", f"Stochastic %K > %D",
            f"ROC = {snap.roc:.3f} (momentum positif)",
        ])
    if bearish_votes >= 3:
        confidence = 55 + bearish_votes * 8
        return AgentSignal(name, Direction.PUT, min(confidence, 95), [
            f"RSI = {snap.rsi:.1f}", f"MACD < signal", f"Stochastic %K < %D",
            f"ROC = {snap.roc:.3f} (momentum négatif)",
        ])
    return _no_trade(name, f"Momentum mitigé ({bullish_votes} bullish / {bearish_votes} bearish sur 4).")


# --- Agent 2 : Trend Following ---------------------------------------------
def _swing_structure(candles: list[Candle]) -> str:
    """Approxime la structure HH/HL (haussière) ou LH/LL (baissière) en
    comparant les extrêmes de la première et de la deuxième moitié de la
    fenêtre récente."""
    if len(candles) < 6:
        return "indéterminée"
    mid = len(candles) // 2
    first_half, second_half = candles[:mid], candles[mid:]
    first_high, first_low = max(c.high for c in first_half), min(c.low for c in first_half)
    second_high, second_low = max(c.high for c in second_half), min(c.low for c in second_half)

    if second_high > first_high and second_low > first_low:
        return "HH/HL (structure haussière)"
    if second_high < first_high and second_low < first_low:
        return "LH/LL (structure baissière)"
    return "structure mixte"


def agent_trend_following(snap: IndicatorSnapshot, recent_candles: list[Candle]) -> AgentSignal:
    name = "Agent 2 — Trend Following"
    if None in (snap.ema9, snap.ema21, snap.ema50, snap.adx):
        return _no_trade(name, "Données insuffisantes.")

    structure = _swing_structure(recent_candles)
    bullish_trend = snap.ema9 > snap.ema21 > snap.ema50
    bearish_trend = snap.ema9 < snap.ema21 < snap.ema50
    strong = snap.adx >= 20

    if bullish_trend and strong and "haussière" in structure:
        return AgentSignal(name, Direction.CALL, 80, [
            "EMA9 > EMA21 > EMA50", f"ADX = {snap.adx:.1f} (tendance forte)", structure,
        ])
    if bearish_trend and strong and "baissière" in structure:
        return AgentSignal(name, Direction.PUT, 80, [
            "EMA9 < EMA21 < EMA50", f"ADX = {snap.adx:.1f} (tendance forte)", structure,
        ])
    if bullish_trend and strong:
        return AgentSignal(name, Direction.CALL, 60, ["EMA alignées haussier", f"ADX = {snap.adx:.1f}", structure])
    if bearish_trend and strong:
        return AgentSignal(name, Direction.PUT, 60, ["EMA alignées baissier", f"ADX = {snap.adx:.1f}", structure])
    return _no_trade(name, f"Tendance pas assez nette (ADX = {snap.adx:.1f}, {structure}).")


# --- Agent 3 : Price Action / Chandeliers -----------------------------------
def agent_price_action(candle: CandleAnalysis) -> AgentSignal:
    name = "Agent 3 — Price Action / Chandeliers"
    bullish = [p for p in candle.patterns if p in REVERSAL_BULLISH_PATTERNS or "Rejection Candle (support)" in p or "Breakout Candle (haussier)" in p]
    bearish = [p for p in candle.patterns if p in REVERSAL_BEARISH_PATTERNS or "Rejection Candle (résistance)" in p or "Breakout Candle (baissier)" in p]

    ratio = candle.geometry.body_wick_ratio
    ratio_note = f"corps/mèches = {ratio:.2f}" if ratio is not None else "corps/mèches = fort (mèches quasi nulles)"

    if bullish and not bearish:
        return AgentSignal(name, Direction.CALL, min(60 + len(bullish) * 10, 90), bullish + [ratio_note])
    if bearish and not bullish:
        return AgentSignal(name, Direction.PUT, min(60 + len(bearish) * 10, 90), bearish + [ratio_note])
    if bullish and bearish:
        return _no_trade(name, f"Patterns contradictoires : {bullish} vs {bearish}.")
    return _no_trade(name, "Aucun pattern exploitable sur cette bougie.")


# --- Agent 4 : Volatilité / Structure ---------------------------------------
def agent_volatility_structure(snap: IndicatorSnapshot, candle: CandleAnalysis) -> AgentSignal:
    name = "Agent 4 — Volatilité / Structure"
    if None in (snap.atr, snap.bb_upper, snap.bb_lower, snap.adx):
        return _no_trade(name, "Données insuffisantes.")

    bb_width_pct = (snap.bb_upper - snap.bb_lower) / snap.close * 100 if snap.close else 0

    if bb_width_pct < 0.15:
        return _no_trade(name, f"Compression forte (bandes de Bollinger très serrées, {bb_width_pct:.2f}%) — signal trop incertain.")
    if snap.adx < 15:
        return _no_trade(name, f"Marché en range (ADX = {snap.adx:.1f}) — pas de structure exploitable.")

    breakout_up = "Breakout Candle (haussier)" in candle.patterns
    breakout_down = "Breakout Candle (baissier)" in candle.patterns

    if breakout_up:
        return AgentSignal(name, Direction.CALL, 70, [
            f"Expansion de volatilité (bandes à {bb_width_pct:.2f}%)", "Breakout Candle haussier confirmé",
        ])
    if breakout_down:
        return AgentSignal(name, Direction.PUT, 70, [
            f"Expansion de volatilité (bandes à {bb_width_pct:.2f}%)", "Breakout Candle baissier confirmé",
        ])
    return _no_trade(name, "Volatilité correcte mais pas de breakout net — cet agent reste prudent par nature.")


# --- Agent 5 : Smart Technical Analyst (généraliste) ------------------------
def agent_smart_analyst(snap: IndicatorSnapshot, candle: CandleAnalysis, recent_candles: list[Candle]) -> AgentSignal:
    name = "Agent 5 — Smart Technical Analyst"
    if None in (snap.ema9, snap.ema21, snap.rsi, snap.macd, snap.macd_signal, snap.adx):
        return _no_trade(name, "Données insuffisantes.")

    score = 0
    reasons = []

    if snap.ema9 > snap.ema21:
        score += 1; reasons.append("EMA9 > EMA21")
    elif snap.ema9 < snap.ema21:
        score -= 1; reasons.append("EMA9 < EMA21")

    if snap.macd > snap.macd_signal:
        score += 1; reasons.append("MACD > signal")
    elif snap.macd < snap.macd_signal:
        score -= 1; reasons.append("MACD < signal")

    if snap.rsi > 55:
        score += 1; reasons.append(f"RSI = {snap.rsi:.1f} (> 55)")
    elif snap.rsi < 45:
        score -= 1; reasons.append(f"RSI = {snap.rsi:.1f} (< 45)")

    bullish_patterns = [p for p in candle.patterns if p in REVERSAL_BULLISH_PATTERNS]
    bearish_patterns = [p for p in candle.patterns if p in REVERSAL_BEARISH_PATTERNS]
    if bullish_patterns:
        score += 1; reasons.append(f"Pattern(s) haussier(s) : {', '.join(bullish_patterns)}")
    if bearish_patterns:
        score -= 1; reasons.append(f"Pattern(s) baissier(s) : {', '.join(bearish_patterns)}")

    structure = _swing_structure(recent_candles)
    if "haussière" in structure:
        score += 1; reasons.append(structure)
    elif "baissière" in structure:
        score -= 1; reasons.append(structure)

    trend_weight = 1.3 if snap.adx >= 25 else 1.0
    weighted = score * trend_weight

    if weighted >= 3:
        return AgentSignal(name, Direction.CALL, min(60 + int(weighted * 8), 92), reasons)
    if weighted <= -3:
        return AgentSignal(name, Direction.PUT, min(60 + int(abs(weighted) * 8), 92), reasons)
    return _no_trade(name, f"Score global insuffisant ({weighted:.1f}/5 facteurs) pour une conviction nette.")


def run_all_agents(
    snap_m5: IndicatorSnapshot,
    candle_m5: CandleAnalysis,
    recent_candles_m5: list[Candle],
) -> list[AgentSignal]:
    """Exécute les 5 agents indépendants sur le même jeu de données M5."""
    return [
        agent_momentum(snap_m5, candle_m5),
        agent_trend_following(snap_m5, recent_candles_m5),
        agent_price_action(candle_m5),
        agent_volatility_structure(snap_m5, candle_m5),
        agent_smart_analyst(snap_m5, candle_m5, recent_candles_m5),
    ]
