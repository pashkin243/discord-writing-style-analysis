import json
import argparse
import numpy as np
from pathlib import Path
from adjustText import adjust_text
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

def build_feature_names(word_vectorizer, char_vectorizer) -> list[str]:
    feature_names = []

    if word_vectorizer is not None:
        word_features = word_vectorizer.get_feature_names_out()
        feature_names.extend([f"[W] {f}" for f in word_features])

    if char_vectorizer is not None:
        char_features = char_vectorizer.get_feature_names_out()
        feature_names.extend([f"[C] {f}" for f in char_features])

    return feature_names

def print_top_component_features(pca, feature_names, top_n=20):
    for comp_idx, component in enumerate(pca.components_[:2], start=1):
        print(f"\n=== Top positive features for PC{comp_idx} ===")
        top_pos = np.argsort(component)[-top_n:][::-1]
        for idx in top_pos:
            print(f"{feature_names[idx]} ({component[idx]:.4f})")

        print(f"\n=== Top negative features for PC{comp_idx} ===")
        top_neg = np.argsort(component)[:top_n]
        for idx in top_neg:
            print(f"{feature_names[idx]} ({component[idx]:.4f})")


def build_features(feature_type, X_train_text, X_val_text, X_test_text):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--features", choices=["word", "char", "both"], default="both")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train))
    val_rows = load_jsonl(Path(args.val))
    test_rows = load_jsonl(Path(args.test))

    X_train_text, y_train = extract_xy(train_rows)
    X_val_text, y_val = extract_xy(val_rows)
    X_test_text, y_test = extract_xy(test_rows)

    X_train, X_val, X_test, word_vectorizer, char_vectorizer = build_features(
        args.features,
        X_train_text,
        X_val_text,
        X_test_text,
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)

    user_vectors = model.coef_
    labels = list(model.classes_)
    feature_names = build_feature_names(word_vectorizer, char_vectorizer)

    if args.method == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(user_vectors)

        explained = reducer.explained_variance_ratio_
        print("\n=== PCA explained variance ===")
        print(f"PC1: {explained[0]:.4f}")
        print(f"PC2: {explained[1]:.4f}")
        print(f"Total: {(explained[0] + explained[1]):.4f}")

        print_top_component_features(reducer, feature_names, top_n=15)
    else:
        reducer = TSNE(n_components=2, perplexity=5, random_state=42)
        reduced = reducer.fit_transform(user_vectors)

    centroid_full = user_vectors.mean(axis=0)
    full_distances = np.linalg.norm(user_vectors - centroid_full, axis=1)

    print("\n=== Style uniqueness (full-space distance from centroid) ===")
    ranking = sorted(zip(labels, full_distances), key=lambda x: x[1], reverse=True)
    for label, score in ranking:
        print(f"{label}: {score:.4f}")

    centroid_2d = reduced.mean(axis=0)
    pca_distances = np.linalg.norm(reduced - centroid_2d, axis=1)

    print("\n=== Style uniqueness (PCA 2D distance from centroid) ===")
    ranking_2d = sorted(zip(labels, pca_distances), key=lambda x: x[1], reverse=True)
    for label, score in ranking_2d:
        print(f"{label}: {score:.4f}")

    plt.figure(figsize=(12, 9))

    texts = []

    for i, label in enumerate(labels):
        x, y = reduced[i]
        plt.scatter(x, y, s=60)
        texts.append(
            plt.text(
                x, y, label,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5)
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    )

    plt.title(f"User Style Map ({args.features}, {args.method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)

    output_path = f"user_map_{args.features}_{args.method}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved 2D visualization to {output_path}")


if __name__ == "__main__":
    main()

# Kasutatud AI