import os
import asyncpg

_pool: asyncpg.Pool | None = None

# SQL schema
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id BIGINT PRIMARY KEY,
    guild_id BIGINT,
    channel_id BIGINT NOT NULL,
    author_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
);

CREATE INDEX IF NOT EXISTS idx_messages_channel_id ON messages(channel_id);
"""

# init_db()
async def init_db() -> None:
    global _pool

    db_url = os.getenv("DATABASE_URL") # railway muutuja
    _pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=5)

    async with _pool.acquire() as conn:
        await conn.execute(CREATE_SQL)

# sõnumi sisestamine db-sse
async def insert_message(
        *,
        message_id: int,
        guild_id: int | None,
        channel_id: int,
        author_id: int,
        content: str,
        created_at
) -> bool:
    
    assert _pool is not None, "DB not initialized"
    async with _pool.acquire() as conn:
        result = await conn.execute(
            """
            INSERT INTO messages(message_id, guild_id, channel_id, author_id, content, created_at, is_backfill)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (message_id) DO NOTHING
            """,
            message_id,
            guild_id,
            channel_id,
            author_id,
            content,
            created_at,
        )
        return result.endswith("1")
    
async def count_messages(channel_id: int) -> int:
    assert _pool is not None, "DB not initialized"
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM messages WHERE channel_id = $1",
            channel_id
        )