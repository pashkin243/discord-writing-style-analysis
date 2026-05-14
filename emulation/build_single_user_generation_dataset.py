import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def normalize_text(text: str) -> str:
    text = (text or "").replace("\n", " ")
    return " ".join(text.split()).strip()


def make_line(author_label: str, text: str) -> str:
    return f"<{author_label}> {normalize_text(text)}"


def split_by_time(rows: list[dict], train_ratio: float = 0.70, val_ratio: float = 0.15):
    rows = sorted(rows, key=lambda r: r["timestamp"])
    n = len(rows)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end == 0 and n >= 1:
        train_end = 1
    if val_end < train_end:
        val_end = train_end

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    return train_rows, val_rows, test_rows


def rows_to_lines(rows: list[dict], author_label: str) -> list[str]:
    lines = []
    for row in rows:
        text = normalize_text(row["content"])
        if not text:
            continue
        lines.append(make_line(author_label, text))
    return lines


def build_single_user_generation_dataset(
    input_path: str,
    output_dir: str,
    author_label: str,
    min_messages: int = 500,
    max_messages: int | None = 5000,
) -> None:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    rows = [r for r in rows if r["author_label"] == author_label]

    if max_messages is not None:
        rows = sorted(rows, key=lambda r: r["timestamp"])[-max_messages:]

    if len(rows) < min_messages:
        raise RuntimeError(
            f"Not enough messages for {author_label}. "
            f"Found {len(rows)}, need at least {min_messages}."
        )

    train_rows, val_rows, test_rows = split_by_time(rows)

    rng = random.Random(42)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    train_lines = rows_to_lines(train_rows, author_label)
    val_lines = rows_to_lines(val_rows, author_label)
    test_lines = rows_to_lines(test_rows, author_label)

    write_lines(output_dir / "generation_train.txt", train_lines)
    write_lines(output_dir / "generation_val.txt", val_lines)
    write_lines(output_dir / "generation_test.txt", test_lines)

    summary = {
        "author_label": author_label,
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "min_messages": min_messages,
        "max_messages": max_messages,
    }

    with (output_dir / "generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Single-user generation dataset built.")
    print(json.dumps(summary, indent=2))