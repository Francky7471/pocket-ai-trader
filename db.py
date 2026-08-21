"""
Pocket AI Trader — Base de données (Phase 3)
==============================================

Connexion à PostgreSQL (hébergé gratuitement sur Neon) via asyncpg.
Ce module s'occupe de :
- créer les tables au démarrage si elles n'existent pas ;
- charger/sauvegarder les réglages utilisateur (persistants entre les
  redémarrages du service, contrairement à la mémoire seule de la Phase 2) ;
- journaliser les propositions de trade et leurs résultats (utilisé à partir
  de la Phase 13, mais le schéma est posé dès maintenant).

Schéma simplifié de la Phase 1, complété au fil des phases suivantes.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import asyncpg

logger = logging.getLogger("pocket_ai_trader.db")

DATABASE_URL = os.environ["DATABASE_URL"]

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    return _pool


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    telegram_id BIGINT PRIMARY KEY,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS settings (
    telegram_id BIGINT PRIMARY KEY REFERENCES users(telegram_id),
    mode TEXT NOT NULL DEFAULT 'demo',
    lot_size DOUBLE PRECISION NOT NULL DEFAULT 0.02,
    take_profit_usd DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    stop_loss_usd DOUBLE PRECISION,
    instruments TEXT[] NOT NULL DEFAULT ARRAY['EURUSD'],
    timeframes TEXT[] NOT NULL DEFAULT ARRAY['H1', 'M15', 'M5'],
    min_confidence_score INT NOT NULL DEFAULT 90,
    max_consecutive_losses INT NOT NULL DEFAULT 2,
    cooldown_minutes INT NOT NULL DEFAULT 30,
    max_daily_loss DOUBLE PRECISION,
    max_daily_trades INT,
    max_drawdown DOUBLE PRECISION,
    daily_profit_target DOUBLE PRECISION,
    scanner_running BOOLEAN NOT NULL DEFAULT FALSE,
    consecutive_losses INT NOT NULL DEFAULT 0,
    cooldown_until TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trade_proposals (
    id SERIAL PRIMARY KEY,
    telegram_id BIGINT NOT NULL REFERENCES users(telegram_id),
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL,
    strategy TEXT NOT NULL,
    score INT NOT NULL,
    reasons TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    timeframes_aligned TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    lot_size DOUBLE PRECISION NOT NULL,
    take_profit_usd DOUBLE PRECISION NOT NULL,
    stop_loss_usd DOUBLE PRECISION,
    risk_level TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    proposal_id INT REFERENCES trade_proposals(id),
    telegram_id BIGINT NOT NULL REFERENCES users(telegram_id),
    result TEXT NOT NULL DEFAULT 'pending',  -- 'win' | 'loss' | 'pending'
    pnl_usd DOUBLE PRECISION,
    closed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS risk_events (
    id SERIAL PRIMARY KEY,
    telegram_id BIGINT NOT NULL REFERENCES users(telegram_id),
    event_type TEXT NOT NULL,  -- 'cooldown' | 'max_loss' | 'emergency_stop'
    details TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def init_db() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA)
    logger.info("Base de données initialisée (tables créées si besoin).")


@dataclass
class DbSettings:
    telegram_id: int
    mode: str = "demo"
    lot_size: float = 0.02
    take_profit_usd: float = 5.0
    stop_loss_usd: Optional[float] = None
    instruments: list = None
    timeframes: list = None
    min_confidence_score: int = 90
    max_consecutive_losses: int = 2
    cooldown_minutes: int = 30
    max_daily_loss: Optional[float] = None
    max_daily_trades: Optional[int] = None
    max_drawdown: Optional[float] = None
    daily_profit_target: Optional[float] = None
    scanner_running: bool = False
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None

    def __post_init__(self):
        if self.instruments is None:
            self.instruments = ["EURUSD"]
        if self.timeframes is None:
            self.timeframes = ["H1", "M15", "M5"]


async def ensure_user(telegram_id: int, is_admin: bool = False) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (telegram_id, is_admin)
            VALUES ($1, $2)
            ON CONFLICT (telegram_id) DO NOTHING
            """,
            telegram_id,
            is_admin,
        )
        await conn.execute(
            """
            INSERT INTO settings (telegram_id)
            VALUES ($1)
            ON CONFLICT (telegram_id) DO NOTHING
            """,
            telegram_id,
        )


async def load_settings(telegram_id: int) -> DbSettings:
    await ensure_user(telegram_id)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM settings WHERE telegram_id = $1", telegram_id
        )
    data = dict(row)
    data.pop("updated_at", None)
    return DbSettings(**data)


async def save_settings(s: DbSettings) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE settings SET
                mode = $2,
                lot_size = $3,
                take_profit_usd = $4,
                stop_loss_usd = $5,
                instruments = $6,
                timeframes = $7,
                min_confidence_score = $8,
                max_consecutive_losses = $9,
                cooldown_minutes = $10,
                max_daily_loss = $11,
                max_daily_trades = $12,
                max_drawdown = $13,
                daily_profit_target = $14,
                scanner_running = $15,
                consecutive_losses = $16,
                cooldown_until = $17,
                updated_at = now()
            WHERE telegram_id = $1
            """,
            s.telegram_id,
            s.mode,
            s.lot_size,
            s.take_profit_usd,
            s.stop_loss_usd,
            s.instruments,
            s.timeframes,
            s.min_confidence_score,
            s.max_consecutive_losses,
            s.cooldown_minutes,
            s.max_daily_loss,
            s.max_daily_trades,
            s.max_drawdown,
            s.daily_profit_target,
            s.scanner_running,
            s.consecutive_losses,
            s.cooldown_until,
        )


async def log_trade_proposal(telegram_id: int, proposal) -> int:
    """proposal : instance de TradeProposal (défini dans bot.py)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO trade_proposals
                (telegram_id, instrument, direction, strategy, score, reasons,
                 timeframes_aligned, lot_size, take_profit_usd, stop_loss_usd, risk_level)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
            """,
            telegram_id,
            proposal.instrument,
            proposal.direction,
            proposal.strategy,
            proposal.score,
            proposal.reasons,
            proposal.timeframes_aligned,
            proposal.lot_size,
            proposal.take_profit_usd,
            proposal.stop_loss_usd,
            proposal.risk_level,
        )
    return row["id"]


async def log_trade_result(proposal_id: int, telegram_id: int, result: str, pnl_usd: Optional[float]) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO trades (proposal_id, telegram_id, result, pnl_usd, closed_at)
            VALUES ($1, $2, $3, $4, now())
            """,
            proposal_id,
            telegram_id,
            result,
            pnl_usd,
        )


async def get_stats(telegram_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE result = 'win') AS wins,
                COUNT(*) FILTER (WHERE result = 'loss') AS losses,
                COALESCE(SUM(pnl_usd), 0) AS total_pnl
            FROM trades
            WHERE telegram_id = $1 AND result != 'pending'
            """,
            telegram_id,
        )
    return dict(row)
