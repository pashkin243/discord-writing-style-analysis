import json
import argparse
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

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ehitab sõnumitest suuremad üksused (chunkid)
def build_chunks_for_author(
    author_rows: list[dict],
    min_chars: int,
    max_chars: int,
) -> list[dict]:
    chunks = []
    current_messages = []
    current_length = 0

    for row in author_rows:
        text = (row["content"] or "").strip()
        if not text:
            continue

        text_len = len(text)
        if current_messages and current_length >= min_chars and (current_length + 1 + text_len) > max_chars:
            chunks.append(make_chunk(current_messages))
            current_messages = []
            current_length = 0

        current_messages.append(row)
        current_length += text_len + 1 

    if current_messages:
        if current_length >= min_chars or not chunks:
            chunks.append(make_chunk(current_messages))

    return chunks

def make_chunk(rows: list[dict]) -> dict:
    text = "\n".join(row["content"] for row in rows)

    return {
        "author_label": rows[0]["author_label"],
        "channel_labels": sorted({row["channel_label"] for row in rows}),
        "start_timestamp": rows[0]["timestamp"],
        "end_timestamp": rows[-1]["timestamp"],
        "message_count": len(rows),
        "text": text,
    }

# splitib tükid treening, valideerimis ja testandmeteks
def split_by_time(author_chunks: list[dict], train_ratio: float, val_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(author_chunks)

    if n == 0:
        return [], [], []

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end == 0 and n >= 1:
        train_end = 1
    if val_end < train_end:
        val_end = train_end

    train = author_chunks[:train_end]
    val = author_chunks[train_end:val_end]
    test = author_chunks[val_end:]

    return train, val, test

def add_sample_ids(rows: list[dict], prefix: str) -> list[dict]:
    output = []
    for i, row in enumerate(rows, start=1):
        row = dict(row)
        row["sample_id"] = f"{prefix}_{i:07d}"
        output.append(row)
    return output

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to filtered JSONL dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for chunked train/val/test files"
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=300,
        help="Minimum characters per chunk"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="Maximum characters per chunk"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--min-chunks-per-author",
        type=int,
        default=5,
        help="Skip authors with fewer than this many chunks"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    if not rows:
        print("No rows found in input dataset")
        return

    by_author = defaultdict(list)
    for row in rows:
        by_author[row["author_label"]].append(row)

    all_train = []
    all_val = []
    all_test = []

    kept_authors = 0
    skipped_authors = 0

    for author_label, author_rows in by_author.items():
        author_rows.sort(key=lambda r: r["timestamp"])

        chunks = build_chunks_for_author(
            author_rows=author_rows,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )

        if len(chunks) < args.min_chunks_per_author:
            skipped_authors += 1
            continue

        kept_authors += 1

        train, val, test = split_by_time(
            author_chunks=chunks,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

    all_train.sort(key=lambda r: (r["author_label"], r["start_timestamp"]))
    all_val.sort(key=lambda r: (r["author_label"], r["start_timestamp"]))
    all_test.sort(key=lambda r: (r["author_label"], r["start_timestamp"]))

    all_train = add_sample_ids(all_train, "TRAIN")
    all_val = add_sample_ids(all_val, "VAL")
    all_test = add_sample_ids(all_test, "TEST")

    train_path = output_dir / "author_chunks_train.jsonl"
    val_path = output_dir / "author_chunks_val.jsonl"
    test_path = output_dir / "author_chunks_test.jsonl"

    write_jsonl(train_path, all_train)
    write_jsonl(val_path, all_val)
    write_jsonl(test_path, all_test)

    print(f"Input messages: {len(rows)}")
    print(f"Authors kept: {kept_authors}")
    print(f"Authors skipped: {skipped_authors}")
    print(f"Train chunks: {len(all_train)}")
    print(f"Val chunks: {len(all_val)}")
    print(f"Test chunks: {len(all_test)}")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    main()

# Kasutatud AI