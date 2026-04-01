import json
from pathlib import Path
from collections import defaultdict


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


def make_line(author_label: str, text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split()).strip()
    return f"<{author_label}> {text}"


def split_by_time_per_user(rows):
    by_author = defaultdict(list)
    for row in rows:
        by_author[row["author_label"]].append(row)

    train, val, test = [], [], []

    for author, author_rows in by_author.items():
        author_rows.sort(key=lambda r: r["timestamp"])
        n = len(author_rows)

        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train.extend(author_rows[:train_end])
        val.extend(author_rows[train_end:val_end])
        test.extend(author_rows[val_end:])

    return train, val, test


def rows_to_lines(rows):
    lines = []
    for row in rows:
        text = (row["content"] or "").strip()
        if not text:
            continue
        lines.append(make_line(row["author_label"], text))
    return lines


def build_generation_dataset(input_path: str, output_dir: str):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)

    train, val, test = split_by_time_per_user(rows)

    import random
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    write_lines(output_dir / "generation_train.txt", rows_to_lines(train))
    write_lines(output_dir / "generation_val.txt", rows_to_lines(val))
    write_lines(output_dir / "generation_test.txt", rows_to_lines(test))

    print("Generation dataset built.")