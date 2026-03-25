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

def filter_full(rows: list[dict]) -> list[dict]:
    return rows

def filter_recent_by_count(rows: list[dict], recent_count: int) -> list[dict]:
    by_author = defaultdict(list)
    for row in rows:
        by_author[row["author_label"]].append(row)
    kept_rows = []
    for author_label, author_rows in by_author.items():
        author_rows.sort(key=lambda r: r["timestamp"])
        kept_rows.extend(author_rows[-recent_count:])
    kept_rows.sort(key=lambda r: r["timestamp"])
    return kept_rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="exports/messages_dataset.jsonl",
        help="Path to exported JSONL dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "recent"],
        required=True,
        help="Filtering mode"
    )
    parser.add_argument(
        "--recent-count",
        type=int,
        default=5000,
        help="Number of most recent messages to keep per user in recent mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path"
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(input_path)
    if not rows:
        print("No rows found in input dataset")
        return
    if args.mode == "full":
        filtered = filter_full(rows)
    else:
        filtered = filter_recent_by_count(rows, args.recent_count)
    write_jsonl(output_path, filtered)

    print(f"Input rows: {len(rows)}")
    print(f"Output rows: {len(filtered)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()