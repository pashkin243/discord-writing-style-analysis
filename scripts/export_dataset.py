import os
import re
import json
import asyncio
from pathlib import Path
import asyncpg

OUTPUT_DIR = Path("exports")
OUTPUT_DIR.mkdir(exist_ok=True)
DATASET_PATH = OUTPUT_DIR / "messages_dataset.jsonl"
USER_MAP_PATH = OUTPUT_DIR / "user_mapping.json"
CHANNEL_MAP_PATH = OUTPUT_DIR / "channel_mapping.json"

# stabiilsed pseudonüümid ID-de põhjal
def pseudonym_map(values: list[int], prefix: str) -> dict[int, str]:
    unique_sorted = sorted(set(values))
    width = max(3, len(str(len(unique_sorted))))
    return {
        value: f"{prefix}_{i:0{width}d}"
        for i, value in enumerate(unique_sorted, start=1)
    }

# @mentionid asendatakse pseudonüümiga
def replace_mentions(text: str, user_map: dict[int, str]) -> str:
    def repl(match: re.Match) -> str:
        user_id = int(match.group(1))
        pseudo = user_map.get(user_id)
        return f"@{pseudo}" if pseudo else "@USER_UNKNOWN"
    return re.sub(r"<@!?(\d+)>", repl, text)

# teatud nimed asendatakse 
def replace_known_names(text: str, name_map: dict[str, str]) -> str:
    if not name_map:
        return text

    for name in sorted(name_map.keys(), key=len, reverse=True):
        escaped = re.escape(name)
        pattern = rf"(?i)\b{escaped}\b"
        text = re.sub(pattern, name_map[name], text)

    return text

def sanitize_text(
    text: str,
    user_map: dict[int, str],
    known_name_map: dict[str, str],
) -> str:
    text = replace_mentions(text, user_map)
    text = replace_known_names(text, known_name_map)
    return text


async def fetch_messages(conn: asyncpg.Connection) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT message_id, author_id, channel_id, content, created_at
        FROM messages
        WHERE content IS NOT NULL
          AND length(trim(content)) > 0
        ORDER BY created_at ASC
        """
    )

# abifunktsioon 
async def fetch_known_user_names(
    conn: asyncpg.Connection,
    user_ids: list[int],
) -> dict[int, list[str]]:
    return {user_id: [] for user_id in user_ids}


async def main() -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")

    conn = await asyncpg.connect(dsn=db_url)

    try:
        rows = await fetch_messages(conn)
        if not rows:
            print("No messages found in database")
            return

        author_ids = [int(row["author_id"]) for row in rows]
        channel_ids = [int(row["channel_id"]) for row in rows]
        user_map = pseudonym_map(author_ids, "USER")
        channel_map = pseudonym_map(channel_ids, "CHANNEL")
        known_names_by_user = await fetch_known_user_names(conn, list(user_map.keys()))

        known_name_map: dict[str, str] = {}
        for user_id, names in known_names_by_user.items():
            pseudo = user_map[user_id]
            for name in names:
                clean = (name or "").strip()
                if clean:
                    known_name_map[clean] = pseudo

        with DATASET_PATH.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(rows, start=1):
                author_id = int(row["author_id"])
                channel_id = int(row["channel_id"])
                content = row["content"]
                created_at = row["created_at"]

                sanitized = sanitize_text(
                    text=content,
                    user_map=user_map,
                    known_name_map=known_name_map,
                )

                sample = {
                    "sample_id": f"MSG_{idx:07d}",
                    "author_label": user_map[author_id],
                    "channel_label": channel_map[channel_id],
                    "timestamp": created_at.isoformat(),
                    "content": sanitized,
                }

                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        with USER_MAP_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in user_map.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )

        with CHANNEL_MAP_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in channel_map.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Export complete: {DATASET_PATH}")
        print(f"User mapping saved: {USER_MAP_PATH}")
        print(f"Channel mapping saved: {CHANNEL_MAP_PATH}")
        print(f"Total exported samples: {len(rows)}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())