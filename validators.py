"""
Pocket AI Trader — Validateurs (Phase 9)
===========================================

Deux validateurs indépendants qui examinent les sorties des 5 agents (et,
en soutien, des 9 stratégies) avant qu'une opportunité ne devienne une
vraie proposition de trade. Comme prévu en §5-6 de la spec : un validateur
peut rejeter même si tous les agents sont d'accord. Le système de scoring
formel (seuil MINIMUM_CONFIDENCE_SCORE, etc.) arrive en Phase 10 — ici on
pose la logique de confirmation et de contrôle des risques.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agents import AgentSignal
from strategy import Direction, StrategySignal
from indicators import IndicatorSnapshot


class ValidationDecision(str, Enum):
    APPROVED_CALL = "APPROVED_CALL"
    APPROVED_PUT = "APPROVED_PUT"
    REJECTED = "REJECTED"


@dataclass
class ValidationResult:
    validator_name: str
    decision: ValidationDecision
    confidence: int  # 0-100
    reasons: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        icon = {"APPROVED_CALL": "✅🟢", "APPROVED_PUT": "✅🔴", "REJECTED": "❌"}[self.decision.value]
        return f"{icon} {self.validator_name} : {self.decision.value} ({self.confidence}/100)"


# --- Validateur 1 : Confirmation --------------------------------------------
def validate_confirmation(
    agent_signals: list[AgentSignal],
    strategy_signals: Optional[list[StrategySignal]] = None,
) -> ValidationResult:
    name = "Validateur 1 — Confirmation"

    calls = [a for a in agent_signals if a.direction == Direction.CALL]
    puts = [a for a in agent_signals if a.direction == Direction.PUT]

    # L'Agent 4 (Volatilité/Structure) a un droit de veto implicite : s'il
    # dit NO_TRADE pour cause de compression ou de range, c'est un signal
    # de prudence qu'on respecte plutôt que de l'ignorer.
    volatility_agent = next((a for a in agent_signals if "Agent 4" in a.agent_name), None)
    volatility_veto = (
        volatility_agent is not None
        and volatility_agent.direction == Direction.NO_TRADE
        and ("compression" in " ".join(volatility_agent.reasons).lower()
             or "range" in " ".join(volatility_agent.reasons).lower())
    )

    majority_needed = 3  # sur 5 agents

    if volatility_veto:
        return ValidationResult(name, ValidationDecision.REJECTED, 0, [
            "Veto de l'Agent 4 (Volatilité/Structure) : marché en compression ou en range.",
        ])

    if len(calls) >= majority_needed:
        avg_conf = sum(a.confidence for a in calls) // len(calls)
        reasons = [f"{len(calls)}/5 agents en CALL"] + [a.agent_name for a in calls]
        if strategy_signals:
            strat_calls = sum(1 for s in strategy_signals if s.direction == Direction.CALL)
            reasons.append(f"Stratégies en soutien : {strat_calls}/{len(strategy_signals)} en CALL")
        return ValidationResult(name, ValidationDecision.APPROVED_CALL, avg_conf, reasons)

    if len(puts) >= majority_needed:
        avg_conf = sum(a.confidence for a in puts) // len(puts)
        reasons = [f"{len(puts)}/5 agents en PUT"] + [a.agent_name for a in puts]
        if strategy_signals:
            strat_puts = sum(1 for s in strategy_signals if s.direction == Direction.PUT)
            reasons.append(f"Stratégies en soutien : {strat_puts}/{len(strategy_signals)} en PUT")
        return ValidationResult(name, ValidationDecision.APPROVED_PUT, avg_conf, reasons)

    return ValidationResult(name, ValidationDecision.REJECTED, 0, [
        f"Pas de majorité claire ({len(calls)} CALL / {len(puts)} PUT sur 5, minimum requis : {majority_needed})."
    ])


# --- Validateur 2 : Risque / Opportunité ------------------------------------
def validate_risk_opportunity(
    v1_result: ValidationResult,
    agent_signals: list[AgentSignal],
    snap: IndicatorSnapshot,
) -> ValidationResult:
    name = "Validateur 2 — Risque/Opportunité"

    if v1_result.decision == ValidationDecision.REJECTED:
        return ValidationResult(name, ValidationDecision.REJECTED, 0, [
            "Non évalué : le Validateur 1 a déjà rejeté l'opportunité."
        ])

    reasons = []
    risk_penalty = 0

    # Cohérence entre agents : combien sont d'accord avec la décision de V1 ?
    target_direction = Direction.CALL if v1_result.decision == ValidationDecision.APPROVED_CALL else Direction.PUT
    agreeing = sum(1 for a in agent_signals if a.direction == target_direction)
    if agreeing < 4:
        risk_penalty += 10
        reasons.append(f"Cohérence modérée : seulement {agreeing}/5 agents d'accord.")
    else:
        reasons.append(f"Bonne cohérence : {agreeing}/5 agents d'accord.")

    # Risque de retournement : RSI extrême dans le sens opposé au trade proposé
    if snap.rsi is not None:
        if target_direction == Direction.CALL and snap.rsi > 75:
            risk_penalty += 20
            reasons.append(f"RSI = {snap.rsi:.1f} (déjà en surachat) — risque de retournement avant le TP.")
        elif target_direction == Direction.PUT and snap.rsi < 25:
            risk_penalty += 20
            reasons.append(f"RSI = {snap.rsi:.1f} (déjà en survente) — risque de retournement avant le TP.")

    # Volatilité insuffisante pour atteindre le TP dans un délai raisonnable
    if snap.atr is not None and snap.close:
        atr_pct = snap.atr / snap.close * 100
        if atr_pct < 0.03:
            risk_penalty += 15
            reasons.append(f"ATR très faible ({atr_pct:.3f}% du prix) — mouvement probablement insuffisant.")

    # Risque de faux breakout : ADX pas assez fort pour un signal de breakout
    if snap.adx is not None and snap.adx < 18:
        risk_penalty += 10
        reasons.append(f"ADX = {snap.adx:.1f} — tendance faible, risque de faux signal.")

    final_confidence = max(v1_result.confidence - risk_penalty, 0)

    if risk_penalty >= 30 or final_confidence < 50:
        return ValidationResult(name, ValidationDecision.REJECTED, final_confidence, reasons + [
            f"Score final trop bas après pénalités de risque ({final_confidence}/100)."
        ])

    decision = ValidationDecision.APPROVED_CALL if target_direction == Direction.CALL else ValidationDecision.APPROVED_PUT
    return ValidationResult(name, decision, final_confidence, reasons)


@dataclass
class ValidationPipelineResult:
    v1: ValidationResult
    v2: ValidationResult

    @property
    def final_approved(self) -> bool:
        return self.v2.decision != ValidationDecision.REJECTED

    @property
    def final_direction(self) -> Optional[Direction]:
        if self.v2.decision == ValidationDecision.APPROVED_CALL:
            return Direction.CALL
        if self.v2.decision == ValidationDecision.APPROVED_PUT:
            return Direction.PUT
        return None

    def risk_level(self) -> str:
        if self.v2.confidence >= 75:
            return "LOW"
        if self.v2.confidence >= 55:
            return "MEDIUM"
        return "HIGH"


def run_validators(
    agent_signals: list[AgentSignal],
    snap: IndicatorSnapshot,
    strategy_signals: Optional[list[StrategySignal]] = None,
) -> ValidationPipelineResult:
    v1 = validate_confirmation(agent_signals, strategy_signals)
    v2 = validate_risk_opportunity(v1, agent_signals, snap)
    return ValidationPipelineResult(v1=v1, v2=v2)
