import json
import argparse
from pathlib import Path

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        return X_train_word, X_val_word, X_test_word, word_vectorizer, char_vectorizer

    if feature_type == "char":
        return X_train_char, X_val_char, X_test_char, word_vectorizer, char_vectorizer

    X_train = hstack([X_train_word, X_train_char])
    X_val = hstack([X_val_word, X_val_char])
    X_test = hstack([X_test_word, X_test_char])

    return X_train, X_val, X_test, word_vectorizer, char_vectorizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to train JSONL")
    parser.add_argument("--val", type=str, required=True, help="Path to val JSONL")
    parser.add_argument("--test", type=str, required=True, help="Path to test JSONL")
    parser.add_argument(
        "--features",
        type=str,
        choices=["word", "char", "both"],
        default="word",
        help="Which TF-IDF features to use",
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
    print(f"Unique authors in train: {len(set(y_train))}")
    print(f"Feature mode: {args.features}")

    X_train, X_val, X_test, word_vectorizer, char_vectorizer = build_features(
        args.features,
        X_train_text,
        X_val_text,
        X_test_text,
    )

    print(f"Final feature count: {X_train.shape[1]}")
    if word_vectorizer is not None:
        print(f"Word feature count: {len(word_vectorizer.vocabulary_)}")
    if char_vectorizer is not None:
        print(f"Char feature count: {len(char_vectorizer.vocabulary_)}")

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print("\n=== RESULTS ===")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")

    print("\n=== TEST CLASSIFICATION REPORT ===")
    print(classification_report(y_test, test_preds, digits=4, zero_division=0))

    # CONFUSION MATRIX (RAW)
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, test_preds, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Raw Counts ({args.features})")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_raw_{args.features}.png")
    plt.close()

    # CONFUSION MATRIX
    cm_normalized = confusion_matrix(
        y_test,
        test_preds,
        labels=labels,
        normalize="true"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Row Normalized ({args.features})")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_normalized_{args.features}.png")
    plt.close()

    print(f"\nSaved confusion matrix to confusion_matrix_raw_{args.features}.png")
    print(f"Saved normalized confusion matrix to confusion_matrix_normalized_{args.features}.png")

if __name__ == "__main__":
    main()

# Kasutatud AI