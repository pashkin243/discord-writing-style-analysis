import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

_pool: asyncpg.Pool | None = None

# SQL schema
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id BIGINT PRIMARY KEY,
    guild_id BIGINT,
    channel_id BIGINT NOT NULL,
    author_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
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
            INSERT INTO messages(message_id, guild_id, channel_id, author_id, content, created_at)
            VALUES ($1,$2,$3,$4,$5,$6)
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
    
async def wipe_channel(channel_id: int) -> None:
    assert _pool is not None, "DB not initialized"
    async with _pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM messages WHERE channel_id = $1",
            channel_id
        )
    
async def count_messages(channel_id: int) -> int:
    assert _pool is not None, "DB not initialized"
    async with _pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM messages WHERE channel_id = $1",
            channel_id
        )

# Kanali profiili statistika arvutamine
async def get_profile(channel_id: int, author_id: int | None = None) -> dict | None:
    assert _pool is not None, "DB not initialized"
    async with _pool.acquire() as conn:
        if author_id is None:
            rows = await conn.fetch(
                "SELECT content FROM messages WHERE channel_id = $1",
                channel_id
            )
        else:
            rows = await conn.fetch(
                "SELECT content FROM messages WHERE channel_id = $1 AND author_id = $2",
                channel_id,
                author_id
            )
        if not rows:
            return None
# statistika arvutamine
        messages = [r["content"] for r in rows]

        total_messages = len(messages)
        total_chars = sum(len(m) for m in messages)
        total_words = sum(len(m.split()) for m in messages)
        exclamations = sum(m.count("!") for m in messages)
        questions = sum(m.count("?") for m in messages)
        uppercase_chars = sum(
            sum(1 for c in m if c.isupper())
            for m in messages
        )
        letters = sum(
            sum(1 for c in m if c.isalpha())
            for m in messages
        )
        avg_length = total_chars / total_messages
        avg_words = total_words / total_messages
        uppercase_ratio = (uppercase_chars / letters) if letters > 0 else 0

        return {
            "messages": total_messages,
            "avg_length": avg_length,
            "avg_words": avg_words,
            "exclamations_per_msg": exclamations / total_messages,
            "questions_per_msg": questions / total_messages,
            "uppercase_ratio": uppercase_ratio
        }

# sõnumite fetchimine
async def get_messages(channel_id: int, author_id: int | None = None, limit: int | None = None) -> list[str]:
    assert _pool is not None, "DB not initialized"

    async with _pool.acquire() as conn:
        if author_id is None:
            if limit is None:
                rows = await conn.fetch(
                    """
                    SELECT content
                    FROM messages
                    WHERE channel_id = $1
                    ORDER BY created_at DESC
                    """,
                    channel_id
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT content
                    FROM messages
                    WHERE channel_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    channel_id,
                    limit
                )
        else:
            if limit is None:
                rows = await conn.fetch(
                    """
                    SELECT content
                    FROM messages
                    WHERE channel_id = $1 AND author_id = $2
                    ORDER BY created_at DESC
                    """,
                    channel_id,
                    author_id
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT content
                    FROM messages
                    WHERE channel_id = $1 AND author_id = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    channel_id,
                    author_id,
                    limit
                )

    return [row["content"] for row in rows]

# detailsem info sõnumi kohta
async def get_message_rows(
    channel_id: int,
    author_id: int | None = None,
    limit: int | None = None
) -> list[dict]:
    assert _pool is not None, "DB not initialized"

    async with _pool.acquire() as conn:
        if author_id is None:
            if limit is None:
                rows = await conn.fetch(
                    """
                    SELECT message_id, channel_id, author_id, content, created_at
                    FROM messages
                    WHERE channel_id = $1
                    ORDER BY created_at DESC
                    """,
                    channel_id
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT message_id, channel_id, author_id, content, created_at
                    FROM messages
                    WHERE channel_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    channel_id,
                    limit
                )
        else:
            if limit is None:
                rows = await conn.fetch(
                    """
                    SELECT message_id, channel_id, author_id, content, created_at
                    FROM messages
                    WHERE channel_id = $1 AND author_id = $2
                    ORDER BY created_at DESC
                    """,
                    channel_id,
                    author_id
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT message_id, channel_id, author_id, content, created_at
                    FROM messages
                    WHERE channel_id = $1 AND author_id = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    channel_id,
                    author_id,
                    limit
                )

    return [dict(row) for row in rows]

# Kasutatud AI