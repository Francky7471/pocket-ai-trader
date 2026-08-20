"""
Pocket AI Trader — Bot Telegram (Phase 2)
==========================================

Squelette du bot de contrôle. À ce stade (mode intérimaire sans VPS) :
- Le bot pilote le SCANNER (démarrage/arrêt de l'analyse), pas l'exécution.
- Chaque opportunité validée est envoyée en ALERTE Telegram avec tous les
  éléments nécessaires pour que l'utilisateur place le trade lui-même dans
  l'appli MT5 mobile (instrument, sens, TP, SL, score, raisons).
- Les moteurs d'analyse (indicateurs, agents IA, scoring, backtest) seront
  branchés dans les phases suivantes (4 à 13) via `engine.py` (pas encore
  créé) — ici on prépare uniquement l'interface Telegram et les structures
  de données qu'elle consomme.

Déploiement prévu (mode intérimaire, budget zéro) : Render ou PythonAnywhere,
offre gratuite, tourne indépendamment du téléphone.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pocket_ai_trader")

# ---------------------------------------------------------------------------
# Configuration (variables d'environnement — jamais de secrets en dur)
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ADMIN_TELEGRAM_ID = int(os.environ["ADMIN_TELEGRAM_ID"])  # ton user_id Telegram

router = Router()


# ---------------------------------------------------------------------------
# État utilisateur — en mémoire pour la Phase 2 (remplacé par PostgreSQL
# en Phase 3, structure des tables déjà définie en Phase 1)
# ---------------------------------------------------------------------------

@dataclass
class RiskSettings:
    max_consecutive_losses: int = 2
    cooldown_minutes: int = 30
    max_daily_loss: Optional[float] = None
    max_daily_trades: Optional[int] = None
    max_drawdown: Optional[float] = None
    daily_profit_target: Optional[float] = None


@dataclass
class UserSettings:
    telegram_id: int
    mode: str = "demo"  # "demo" | "real" — informatif en exécution manuelle
    lot_size: float = 0.02
    take_profit_usd: float = 5.0
    stop_loss_usd: Optional[float] = None  # à définir avec l'utilisateur
    instruments: list = field(default_factory=lambda: ["EURUSD"])
    timeframes: list = field(default_factory=lambda: ["H1", "M15", "M5"])
    min_confidence_score: int = 90
    risk: RiskSettings = field(default_factory=RiskSettings)
    scanner_running: bool = False
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None


# clé = telegram_id ; un seul utilisateur pour l'instant (toi), structure
# prête pour plusieurs si besoin un jour
USERS: dict[int, UserSettings] = {}


def get_settings(telegram_id: int) -> UserSettings:
    if telegram_id not in USERS:
        USERS[telegram_id] = UserSettings(telegram_id=telegram_id)
    return USERS[telegram_id]


def is_authorized(telegram_id: int) -> bool:
    return telegram_id == ADMIN_TELEGRAM_ID


# ---------------------------------------------------------------------------
# Structure d'une opportunité de trade (produite par le moteur d'analyse,
# consommée ici pour formatter l'alerte — voir §19/§29/§34 de la spec)
# ---------------------------------------------------------------------------

@dataclass
class TradeProposal:
    instrument: str
    direction: str  # "BUY" | "SELL"
    strategy: str
    score: int  # /100
    reasons: list[str]
    timeframes_aligned: list[str]
    lot_size: float
    take_profit_usd: float
    stop_loss_usd: Optional[float]
    risk_level: str  # "LOW" | "MEDIUM" | "HIGH"


def format_alert(proposal: TradeProposal) -> str:
    reasons_block = "\n".join(f"✓ {r}" for r in proposal.reasons)
    tf_block = " / ".join(proposal.timeframes_aligned)
    sl_line = (
        f"SL suggéré : {proposal.stop_loss_usd:.2f} $"
        if proposal.stop_loss_usd is not None
        else "SL : à définir"
    )
    return (
        "🚨 OPPORTUNITÉ DÉTECTÉE\n\n"
        f"Instrument : {proposal.instrument}\n"
        f"Direction : {proposal.direction}\n"
        f"Stratégie : {proposal.strategy}\n"
        f"Timeframes alignés : {tf_block}\n"
        f"Score : {proposal.score}/100\n"
        f"Lot : {proposal.lot_size}\n"
        f"TP suggéré : {proposal.take_profit_usd:.2f} $\n"
        f"{sl_line}\n"
        f"Risque : {proposal.risk_level}\n\n"
        f"Raisons :\n{reasons_block}\n\n"
        "⚠️ Aucune exécution automatique — place le trade toi-même dans "
        "MT5 mobile avec ces paramètres. Aucune garantie de résultat."
    )


# ---------------------------------------------------------------------------
# Menu principal
# ---------------------------------------------------------------------------

def main_menu() -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(text="🟢 Démarrer scan", callback_data="scan_start"),
            InlineKeyboardButton(text="🔴 Arrêter scan", callback_data="scan_stop"),
        ],
        [
            InlineKeyboardButton(text="💰 Lot / TP", callback_data="settings_amount"),
            InlineKeyboardButton(text="📊 Marchés", callback_data="settings_instruments"),
        ],
        [
            InlineKeyboardButton(text="📈 Statistiques", callback_data="stats"),
            InlineKeyboardButton(text="🧪 Backtest", callback_data="backtest"),
        ],
        [
            InlineKeyboardButton(text="🛡️ Gestion du risque", callback_data="risk"),
            InlineKeyboardButton(text="📜 Historique", callback_data="history"),
        ],
        [
            InlineKeyboardButton(text="🟢 DEMO", callback_data="mode_demo"),
            InlineKeyboardButton(text="🔴 REAL", callback_data="mode_real"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ---------------------------------------------------------------------------
# Commandes
# ---------------------------------------------------------------------------

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    if not is_authorized(message.from_user.id):
        await message.answer("⛔ Accès non autorisé.")
        return
    get_settings(message.from_user.id)
    await message.answer(
        "🤖 Pocket AI Trader\n\n"
        "Mode intérimaire : exécution manuelle. Le bot scanne et t'envoie "
        "des alertes, tu places les trades toi-même dans MT5 mobile.",
        reply_markup=main_menu(),
    )


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    if not is_authorized(message.from_user.id):
        return
    s = get_settings(message.from_user.id)
    state = "🟢 RUNNING" if s.scanner_running else "🔴 STOPPED"
    cooldown = (
        f"\nCooldown jusqu'à : {s.cooldown_until:%H:%M}"
        if s.cooldown_until and s.cooldown_until > datetime.now()
        else ""
    )
    await message.answer(
        f"Bot : {state}\n"
        f"Mode : {s.mode.upper()}\n"
        f"Lot : {s.lot_size} — TP : {s.take_profit_usd:.2f} $\n"
        f"Instruments suivis : {', '.join(s.instruments)}\n"
        f"Timeframes : {', '.join(s.timeframes)}\n"
        f"Seuil de confiance min. : {s.min_confidence_score}%\n"
        f"Pertes consécutives : {s.consecutive_losses}"
        f"{cooldown}"
    )


@router.message(Command("startbot"))
async def cmd_startbot(message: Message) -> None:
    if not is_authorized(message.from_user.id):
        return
    s = get_settings(message.from_user.id)
    if s.cooldown_until and s.cooldown_until > datetime.now():
        await message.answer(
            f"⏸️ Cooldown actif jusqu'à {s.cooldown_until:%H:%M} "
            f"({s.consecutive_losses} pertes consécutives). Scan non démarré."
        )
        return
    s.scanner_running = True
    # TODO Phase 4+ : lancer réellement le scanner (market data → agents →
    # validateurs → scoring → format_alert → envoi Telegram)
    await message.answer("🟢 Scanner démarré. Tu recevras une alerte par opportunité validée.")


@router.message(Command("stopbot"))
async def cmd_stopbot(message: Message) -> None:
    if not is_authorized(message.from_user.id):
        return
    s = get_settings(message.from_user.id)
    s.scanner_running = False
    await message.answer("🛑 BOT ARRÊTÉ\n\nAucune nouvelle alerte ne sera envoyée.")


@router.message(Command("demo"))
async def cmd_demo(message: Message) -> None:
    if not is_authorized(message.from_user.id):
        return
    get_settings(message.from_user.id).mode = "demo"
    await message.answer("🟢 Mode DEMO activé.")


@router.message(Command("real"))
async def cmd_real(message: Message) -> None:
    if not is_authorized(message.from_user.id):
        return
    s = get_settings(message.from_user.id)
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ CONFIRMER", callback_data="confirm_real"),
                InlineKeyboardButton(text="❌ ANNULER", callback_data="cancel_real"),
            ]
        ]
    )
    await message.answer(
        "⚠️ Vous êtes sur le point de passer en mode RÉEL.\n\n"
        f"Lot : {s.lot_size}\n"
        f"TP : {s.take_profit_usd:.2f} $\n"
        "Rappel : exécution manuelle — c'est toi qui places chaque trade.",
        reply_markup=kb,
    )


@router.callback_query(F.data == "confirm_real")
async def confirm_real(callback: CallbackQuery) -> None:
    if not is_authorized(callback.from_user.id):
        return
    get_settings(callback.from_user.id).mode = "real"
    await callback.message.edit_text("🔴 Mode RÉEL activé.")
    await callback.answer()


@router.callback_query(F.data == "cancel_real")
async def cancel_real(callback: CallbackQuery) -> None:
    await callback.message.edit_text("Annulé — mode inchangé.")
    await callback.answer()


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    if not is_authorized(message.from_user.id):
        return
    await message.answer(
        "Commandes disponibles :\n"
        "/start — menu principal\n"
        "/status — état du bot\n"
        "/startbot — démarrer le scanner\n"
        "/stopbot — arrêter le scanner\n"
        "/balance — solde (à connecter en Phase 3)\n"
        "/trades — derniers trades\n"
        "/stats — statistiques\n"
        "/backtest — lancer un backtest\n"
        "/settings — paramètres\n"
        "/risk — gestion du risque\n"
        "/demo — passer en DEMO\n"
        "/real — passer en RÉEL (confirmation requise)\n"
        "/help — cette aide"
    )


# ---------------------------------------------------------------------------
# Callbacks du menu (placeholders — logique complète en phases suivantes)
# ---------------------------------------------------------------------------

@router.callback_query(F.data == "scan_start")
async def cb_scan_start(callback: CallbackQuery) -> None:
    await cmd_startbot(callback.message)  # réutilise la logique
    await callback.answer()


@router.callback_query(F.data == "scan_stop")
async def cb_scan_stop(callback: CallbackQuery) -> None:
    await cmd_stopbot(callback.message)
    await callback.answer()


@router.callback_query(F.data.in_({"settings_amount", "settings_instruments", "stats", "backtest", "risk", "history"}))
async def cb_placeholder(callback: CallbackQuery) -> None:
    await callback.answer("Disponible dans une phase suivante.", show_alert=True)


# ---------------------------------------------------------------------------
# Fonction utilisée par le futur moteur d'analyse pour notifier l'utilisateur
# (à appeler depuis engine.py une fois les phases 4-13 branchées)
# ---------------------------------------------------------------------------

async def send_trade_alert(bot: Bot, telegram_id: int, proposal: TradeProposal) -> None:
    await bot.send_message(telegram_id, format_alert(proposal))


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    logger.info("Pocket AI Trader — bot démarré.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
