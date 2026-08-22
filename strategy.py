"""
Pocket AI Trader — Strategy Engine (Phase 7)
===============================================

Combine les indicateurs (Phase 5) et l'analyse de chandeliers (Phase 6) en
9 stratégies indépendantes : les 8 définies dans la spec initiale (§32) +
la stratégie multi-timeframe H1/M15/M5 (ajoutée au même niveau que les
autres, comme demandé — aucun traitement de faveur).

Chaque stratégie produit un StrategySignal (CALL / PUT / NO_TRADE + score
+ raisons). Ce module ne décide de rien tout seul — les 5 agents IA et les
2 validateurs (Phases 8-9) examineront ces signaux, comme tous les autres.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from indicators import IndicatorSnapshot
from candlestick import CandleAnalysis


class Direction(str, Enum):
    CALL = "CALL"
    PUT = "PUT"
    NO_TRADE = "NO_TRADE"


@dataclass
class StrategySignal:
    strategy_name: str
    direction: Direction
    score: int  # 0-100
    reasons: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        icon = {"CALL": "🟢", "PUT": "🔴", "NO_TRADE": "⚪"}[self.direction.value]
        return f"{icon} {self.strategy_name} : {self.direction.value} ({self.score}/100)"


REVERSAL_BULLISH_PATTERNS = {
    "Hammer (retournement haussier potentiel)",
    "Bullish Engulfing",
    "Morning Star",
    "Pin Bar haussier",
    "Tweezer Bottom",
}
REVERSAL_BEARISH_PATTERNS = {
    "Shooting Star (retournement baissier potentiel)",
    "Bearish Engulfing",
    "Evening Star",
    "Pin Bar baissier",
    "Tweezer Top",
}


def _no_trade(name: str, reason: str) -> StrategySignal:
    return StrategySignal(name, Direction.NO_TRADE, 0, [reason])


# --- Strategy A : Trend + Momentum ---------------------------------------
def strategy_a_trend_momentum(snap: IndicatorSnapshot, candle: CandleAnalysis) -> StrategySignal:
    name = "A — Trend + Momentum"
    if None in (snap.ema9, snap.ema21, snap.ema50, snap.rsi, snap.macd, snap.macd_signal):
        return _no_trade(name, "Données insuffisantes.")

    bullish_trend = snap.ema9 > snap.ema21 > snap.ema50
    bearish_trend = snap.ema9 < snap.ema21 < snap.ema50
    macd_bullish = snap.macd > snap.macd_signal
    momentum_bullish = snap.rsi > 50

    if bullish_trend and macd_bullish and momentum_bullish:
        score = 60 + min(20, int(snap.rsi - 50)) + (10 if snap.macd_hist and snap.macd_hist > 0 else 0)
        return StrategySignal(name, Direction.CALL, min(score, 95), [
            "EMA9 > EMA21 > EMA50 (tendance haussière alignée)",
            "MACD > signal", f"RSI = {snap.rsi:.1f} (> 50)",
        ])
    if bearish_trend and not macd_bullish and not momentum_bullish:
        score = 60 + min(20, int(50 - snap.rsi)) + (10 if snap.macd_hist and snap.macd_hist < 0 else 0)
        return StrategySignal(name, Direction.PUT, min(score, 95), [
            "EMA9 < EMA21 < EMA50 (tendance baissière alignée)",
            "MACD < signal", f"RSI = {snap.rsi:.1f} (< 50)",
        ])
    return _no_trade(name, "Tendance et momentum non alignés.")


# --- Strategy B : Breakout + Volatilité -----------------------------------
def strategy_b_breakout_volatility(snap: IndicatorSnapshot, candle: CandleAnalysis) -> StrategySignal:
    name = "B — Breakout + Volatilité"
    breakout_up = any("Breakout Candle (haussier)" in p for p in candle.patterns)
    breakout_down = any("Breakout Candle (baissier)" in p for p in candle.patterns)

    if not (breakout_up or breakout_down):
        return _no_trade(name, "Aucune bougie de breakout détectée.")
    if snap.adx is None or snap.adx < 20:
        return _no_trade(name, "ADX trop faible pour confirmer la force du mouvement.")

    if breakout_up:
        return StrategySignal(name, Direction.CALL, 70, [
            "Breakout Candle haussier détecté", f"ADX = {snap.adx:.1f} (> 20)",
        ])
    return StrategySignal(name, Direction.PUT, 70, [
        "Breakout Candle baissier détecté", f"ADX = {snap.adx:.1f} (> 20)",
    ])


# --- Strategy C : Reversal + Candlestick ----------------------------------
def strategy_c_reversal_candlestick(snap: IndicatorSnapshot, candle: CandleAnalysis) -> StrategySignal:
    name = "C — Reversal + Candlestick"
    if snap.rsi is None:
        return _no_trade(name, "RSI indisponible.")

    bullish_patterns = [p for p in candle.patterns if p in REVERSAL_BULLISH_PATTERNS]
    bearish_patterns = [p for p in candle.patterns if p in REVERSAL_BEARISH_PATTERNS]

    if bullish_patterns and snap.rsi < 40:
        return StrategySignal(name, Direction.CALL, 65 + len(bullish_patterns) * 5, [
            f"Pattern(s) de retournement haussier : {', '.join(bullish_patterns)}",
            f"RSI = {snap.rsi:.1f} (survente)",
        ])
    if bearish_patterns and snap.rsi > 60:
        return StrategySignal(name, Direction.PUT, 65 + len(bearish_patterns) * 5, [
            f"Pattern(s) de retournement baissier : {', '.join(bearish_patterns)}",
            f"RSI = {snap.rsi:.1f} (surachat)",
        ])
    return _no_trade(name, "Pas de pattern de retournement confirmé par le RSI.")


# --- Strategy D : Support/Resistance + Price Action -----------------------
def strategy_d_support_resistance(snap: IndicatorSnapshot, candle: CandleAnalysis) -> StrategySignal:
    name = "D — Support/Resistance + Price Action"
    rejection_support = "Rejection Candle (support)" in candle.patterns
    rejection_resistance = "Rejection Candle (résistance)" in candle.patterns
    pin_bullish = "Pin Bar haussier" in candle.patterns
    pin_bearish = "Pin Bar baissier" in candle.patterns

    if rejection_support or pin_bullish:
        reasons = [p for p in ["Rejection Candle (support)", "Pin Bar haussier"] if p in candle.patterns]
        return StrategySignal(name, Direction.CALL, 60 + len(reasons) * 10, reasons)
    if rejection_resistance or pin_bearish:
        reasons = [p for p in ["Rejection Candle (résistance)", "Pin Bar baissier"] if p in candle.patterns]
        return StrategySignal(name, Direction.PUT, 60 + len(reasons) * 10, reasons)
    return _no_trade(name, "Aucun rejet de niveau détecté.")


# --- Strategy E : EMA + MACD + RSI ----------------------------------------
def strategy_e_ema_macd_rsi(snap: IndicatorSnapshot, candle: CandleAnalysis) -> StrategySignal:
    name = "E — EMA + MACD + RSI"
    if None in (snap.ema9, snap.ema21, snap.macd, snap.macd_signal, snap.rsi):
        return _no_trade(name, "Données insuffisantes.")

    votes_bullish = sum([snap.ema9 > snap.ema21, snap.macd > snap.macd_signal, snap.rsi > 50])
    votes_bearish = sum([snap.ema9 < snap.ema21, snap.macd < snap.macd_signal, snap.rsi < 50])

    if votes_bullish == 3:
        return StrategySignal(name, Direction.CALL, 80, ["EMA, MACD et RSI tous alignés haussiers"])
    if votes_bearish == 3:
        return StrategySignal(name, Direction.PUT, 80, ["EMA, MACD et RSI tous alignés baissiers"])
    return _no_trade(name, f"Confirmation partielle seulement ({max(votes_bullish, votes_bearish)}/3).")


# --- Strategy F : Bollinger + RSI -----------------------------------------
def strategy_f_bollinger_rsi(snap: IndicatorSnapshot, candle: CandleAnalysis) -> StrategySignal:
    name = "F — Bollinger + RSI"
    if None in (snap.bb_lower, snap.bb_upper, snap.rsi):
        return _no_trade(name, "Données insuffisantes.")

    if snap.close <= snap.bb_lower and snap.rsi < 35:
        return StrategySignal(name, Direction.CALL, 70, [
            "Clôture sous la bande de Bollinger basse", f"RSI = {snap.rsi:.1f} (survente)",
        ])
    if snap.close >= snap.bb_upper and snap.rsi > 65:
        return StrategySignal(name, Direction.PUT, 70, [
            "Clôture au-dessus de la bande de Bollinger haute", f"RSI = {snap.rsi:.1f} (surachat)",
        ])
    return _no_trade(name, "Prix dans le canal de Bollinger, pas d'extrême RSI.")


# --- Strategy G : ADX + EMA -----------------------------------------------
def strategy_g_adx_ema(snap: IndicatorSnapshot, candle: CandleAnalysis) -> StrategySignal:
    name = "G — ADX + EMA"
    if None in (snap.adx, snap.ema9, snap.ema21):
        return _no_trade(name, "Données insuffisantes.")
    if snap.adx < 25:
        return _no_trade(name, f"ADX = {snap.adx:.1f} (< 25, tendance trop faible).")

    if snap.ema9 > snap.ema21:
        return StrategySignal(name, Direction.CALL, 65, [f"ADX = {snap.adx:.1f} (tendance forte)", "EMA9 > EMA21"])
    if snap.ema9 < snap.ema21:
        return StrategySignal(name, Direction.PUT, 65, [f"ADX = {snap.adx:.1f} (tendance forte)", "EMA9 < EMA21"])
    return _no_trade(name, "EMA9 = EMA21, pas de direction claire.")


# --- Strategy H : Multi-confirmation (vote des stratégies A/E/F/G) -------
def strategy_h_multi_confirmation(snap: IndicatorSnapshot, candle: CandleAnalysis) -> StrategySignal:
    name = "H — Multi-confirmation"
    votes = [
        strategy_a_trend_momentum(snap, candle),
        strategy_e_ema_macd_rsi(snap, candle),
        strategy_f_bollinger_rsi(snap, candle),
        strategy_g_adx_ema(snap, candle),
    ]
    calls = [v for v in votes if v.direction == Direction.CALL]
    puts = [v for v in votes if v.direction == Direction.PUT]

    if len(calls) >= 3:
        return StrategySignal(name, Direction.CALL, 75 + len(calls) * 3, [
            f"{len(calls)}/4 sous-stratégies d'accord (CALL)"
        ] + [v.strategy_name for v in calls])
    if len(puts) >= 3:
        return StrategySignal(name, Direction.PUT, 75 + len(puts) * 3, [
            f"{len(puts)}/4 sous-stratégies d'accord (PUT)"
        ] + [v.strategy_name for v in puts])
    return _no_trade(name, f"Pas de majorité claire ({len(calls)} CALL / {len(puts)} PUT sur 4).")


# --- Strategy I : Multi-timeframe H1 / M15 / M5 ---------------------------
def strategy_i_multi_timeframe(
    snap_h1: Optional[IndicatorSnapshot],
    snap_m15: Optional[IndicatorSnapshot],
    snap_m5: Optional[IndicatorSnapshot],
) -> StrategySignal:
    name = "I — Multi-timeframe (H1/M15/M5)"
    if not all([snap_h1, snap_m15, snap_m5]):
        return _no_trade(name, "Un ou plusieurs timeframes indisponibles.")
    if None in (snap_h1.ema50, snap_h1.ema200, snap_m15.rsi, snap_m15.macd, snap_m15.macd_signal,
                snap_m5.ema9, snap_m5.ema21):
        return _no_trade(name, "Données insuffisantes sur un des timeframes.")

    h1_bullish = snap_h1.ema50 > snap_h1.ema200
    h1_bearish = snap_h1.ema50 < snap_h1.ema200
    m15_bullish = snap_m15.rsi > 50 and snap_m15.macd > snap_m15.macd_signal
    m15_bearish = snap_m15.rsi < 50 and snap_m15.macd < snap_m15.macd_signal
    m5_bullish = snap_m5.ema9 > snap_m5.ema21
    m5_bearish = snap_m5.ema9 < snap_m5.ema21

    if h1_bullish and m15_bullish and m5_bullish:
        return StrategySignal(name, Direction.CALL, 85, [
            "H1 : EMA50 > EMA200 (tendance de fond haussière)",
            "M15 : RSI > 50 et MACD > signal (momentum confirmé)",
            "M5 : EMA9 > EMA21 (déclencheur d'entrée)",
        ])
    if h1_bearish and m15_bearish and m5_bearish:
        return StrategySignal(name, Direction.PUT, 85, [
            "H1 : EMA50 < EMA200 (tendance de fond baissière)",
            "M15 : RSI < 50 et MACD < signal (momentum confirmé)",
            "M5 : EMA9 < EMA21 (déclencheur d'entrée)",
        ])
    return _no_trade(name, "Les 3 timeframes ne sont pas alignés dans le même sens.")


ALL_SINGLE_TF_STRATEGIES = [
    strategy_a_trend_momentum,
    strategy_b_breakout_volatility,
    strategy_c_reversal_candlestick,
    strategy_d_support_resistance,
    strategy_e_ema_macd_rsi,
    strategy_f_bollinger_rsi,
    strategy_g_adx_ema,
    strategy_h_multi_confirmation,
]


def run_all_strategies(
    snap_m5: IndicatorSnapshot,
    candle_m5: CandleAnalysis,
    snap_h1: Optional[IndicatorSnapshot] = None,
    snap_m15: Optional[IndicatorSnapshot] = None,
) -> list[StrategySignal]:
    """Exécute les 9 stratégies et renvoie leurs signaux. Les 8 premières
    travaillent sur M5 (là où on exécute) ; la 9e a besoin des 3 timeframes."""
    signals = [fn(snap_m5, candle_m5) for fn in ALL_SINGLE_TF_STRATEGIES]
    signals.append(strategy_i_multi_timeframe(snap_h1, snap_m15, snap_m5))
    return signals
