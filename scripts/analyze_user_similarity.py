import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def extract_xy(rows: list[dict]) -> tuple[list[str], list[str]]:
    texts = [row["text"] for row in rows]
    labels = [row["author_label"] for row in rows]
    return texts, labels

def build_features(
    feature_type: str,
    X_train_text: list[str],
    X_val_text: list[str],
    X_test_text: list[str],
):
    word_vectorizer = None
    char_vectorizer = None

    if feature_type in {"word", "both"}:
        word_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )
        X_train_word = word_vectorizer.fit_transform(X_train_text)
        X_val_word = word_vectorizer.transform(X_val_text)
        X_test_word = word_vectorizer.transform(X_test_text)
    else:
        X_train_word = X_val_word = X_test_word = None

    if feature_type in {"char", "both"}:
        char_vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        )
        X_train_char = char_vectorizer.fit_transform(X_train_text)
        X_val_char = char_vectorizer.transform(X_val_text)
        X_test_char = char_vectorizer.transform(X_test_text)
    else:
        X_train_char = X_val_char = X_test_char = None

    if feature_type == "word":
        return X_train_word, X_val_word, X_test_word

    if feature_type == "char":
        return X_train_char, X_val_char, X_test_char

    X_train = hstack([X_train_word, X_train_char])
    X_val = hstack([X_val_word, X_val_char])
    X_test = hstack([X_test_word, X_test_char])
    return X_train, X_val, X_test

def save_similarity_csv(labels: list[str], sim_matrix, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, label in enumerate(labels):
            row = ",".join(f"{v:.4f}" for v in sim_matrix[i])
            f.write(f"{label},{row}\n")

def print_top_similar_users(labels: list[str], sim_matrix, top_k: int = 5) -> None:
    for i, label in enumerate(labels):
        pairs = []
        for j, other in enumerate(labels):
            if i == j:
                continue
            pairs.append((other, sim_matrix[i, j]))
        pairs.sort(key=lambda x: x[1], reverse=True)

        print(f"\nMost similar users to {label}:")
        for other, score in pairs[:top_k]:
            print(f"  {other}: {score:.4f}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument(
        "--features",
        type=str,
        choices=["word", "char", "both"],
        default="both",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="similarity",
        help="Prefix for saved output files",
    )
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train))
    val_rows = load_jsonl(Path(args.val))
    test_rows = load_jsonl(Path(args.test))

    X_train_text, y_train = extract_xy(train_rows)
    X_val_text, y_val = extract_xy(val_rows)
    X_test_text, y_test = extract_xy(test_rows)

    print(f"Train samples: {len(X_train_text)}")
    print(f"Val samples: {len(X_val_text)}")
    print(f"Test samples: {len(X_test_text)}")
    print(f"Feature mode: {args.features}")

    X_train, X_val, X_test = build_features(
        args.features,
        X_train_text,
        X_val_text,
        X_test_text,
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    labels = list(model.classes_)
    coef_matrix = model.coef_

    # koosinuskaugus
    sim_matrix = cosine_similarity(coef_matrix)
    csv_path = Path(f"{args.prefix}_{args.features}.csv")
    save_similarity_csv(labels, sim_matrix, csv_path)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=labels,
        yticklabels=labels,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
    )
    
    plt.title(f"User Similarity Matrix ({args.features})")
    plt.xlabel("User")
    plt.ylabel("User")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    png_path = Path(f"{args.prefix}_{args.features}.png")
    plt.savefig(png_path)
    plt.close()

    print(f"\nSaved similarity matrix CSV")
    print(f"Saved similarity heatmap")
    print_top_similar_users(labels, sim_matrix, top_k=5)

if __name__ == "__main__":
    main()