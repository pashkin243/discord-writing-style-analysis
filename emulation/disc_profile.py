import re
from collections import Counter

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import db
import style


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_feature_names(word_vectorizer, char_vectorizer) -> list[str]:
    feature_names = []

    if word_vectorizer is not None:
        word_features = word_vectorizer.get_feature_names_out()
        feature_names.extend([f"[W] {f}" for f in word_features])

    if char_vectorizer is not None:
        char_features = char_vectorizer.get_feature_names_out()
        feature_names.extend([f"[C] {f}" for f in char_features])

    return feature_names


def split_feature_names(feature_names: list[str]) -> tuple[list[str], list[str]]:
    word_feats = []
    char_feats = []

    for feat in feature_names:
        if feat.startswith("[W] "):
            word_feats.append(feat[4:])
        elif feat.startswith("[C] "):
            char_feats.append(feat[4:])

    return word_feats, char_feats


def build_features(feature_type: str, texts: list[str]):
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
        X_word = word_vectorizer.fit_transform(texts)
    else:
        X_word = None

    if feature_type in {"char", "both"}:
        char_vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        )
        X_char = char_vectorizer.fit_transform(texts)
    else:
        X_char = None

    if feature_type == "word":
        return X_word, word_vectorizer, char_vectorizer

    if feature_type == "char":
        return X_char, word_vectorizer, char_vectorizer

    X = hstack([X_word, X_char])
    return X, word_vectorizer, char_vectorizer


def get_top_user_features(
    model: LogisticRegression,
    feature_names: list[str],
    target_label: str,
    top_n: int = 20,
) -> tuple[list[str], list[str]]:
    class_index = list(model.classes_).index(target_label)
    coef = model.coef_[class_index]
    top_indices = coef.argsort()[-top_n:][::-1]

    selected = [feature_names[idx] for idx in top_indices]
    return split_feature_names(selected)


def compute_uniqueness_score(
    model: LogisticRegression,
    target_label: str,
) -> float:
    class_index = list(model.classes_).index(target_label)
    user_vectors = model.coef_
    centroid = user_vectors.mean(axis=0)
    return float(((user_vectors[class_index] - centroid) ** 2).sum() ** 0.5)

def select_diverse_messages(scored_messages: list[dict], top_k: int = 10) -> list[dict]:
    selected = []
    seen_endings = set()
    seen_emoji_patterns = set()

    for msg in scored_messages:
        text = msg["text"].strip()
        if not text:
            continue

        ending = text[-3:] if len(text) >= 3 else text
        emojis = tuple(style.extract_all_emojis(text)[-2:])  # last 2 emojis if any

        # avoid too many almost-identical endings / emoji patterns
        if ending in seen_endings and emojis in seen_emoji_patterns:
            continue

        selected.append(msg)
        seen_endings.add(ending)
        seen_emoji_patterns.add(emojis)

        if len(selected) >= top_k:
            break

    return selected

def score_representative_message(text: str, profile: dict) -> float:
    words = tokenize(text)
    word_count = len(words)
    char_count = len(text)

    score = 0.0

    # 1. length closeness
    score -= abs(char_count - profile["style_stats"]["avg_length"]) * 0.04
    score -= abs(word_count - profile["style_stats"]["avg_words"]) * 0.15

    # 2. punctuation closeness
    score -= abs(text.count("!") - profile["style_stats"]["exclamations_per_msg"]) * 0.5
    score -= abs(text.count("?") - profile["style_stats"]["questions_per_msg"]) * 0.5
    score -= abs(text.count(".") - profile["style_stats"]["periods_per_msg"]) * 0.3
    score -= abs(text.count(",") - profile["style_stats"]["commas_per_msg"]) * 0.3
    score -= abs(text.count("\n") - profile["style_stats"]["avg_newlines_per_msg"]) * 0.5

    # 3. common/discriminative word overlap
    target_words = set(profile["top_word_features"])
    common_words = set(words)
    score += len(target_words.intersection(common_words)) * 0.6

    # 4. emoji overlap (reduced weight)
    msg_emojis = set(style.extract_all_emojis(text))
    target_emojis = {emoji for emoji, _ in profile["style_stats"]["common_emojis"]}
    emoji_matches = len(target_emojis.intersection(msg_emojis))
    score += min(emoji_matches, 1) * 0.8

    # 5. discriminative char features (capped influence)
    target_char_feats = set(profile["top_char_features"])
    matched_char_feats = 0
    for feat in target_char_feats:
        if feat and feat in text:
            matched_char_feats += 1
    score += min(matched_char_feats, 4) * 0.35

    # 6. small penalty for very short messages
    if char_count < 8:
        score -= 1.0

    return score

def categorize_message(text: str) -> str:
    emojis = style.extract_all_emojis(text)
    exclamations = text.count("!")
    questions = text.count("?")
    char_count = len(text.strip())

    if char_count <= 25:
        return "short_reactive"

    if len(emojis) >= 1:
        return "emoji_heavy"

    if exclamations + questions >= 3:
        return "punctuation_heavy"

    if char_count >= 120:
        return "longer_expressive"

    return "plain_typical"

def build_message_buckets(scored_messages: list[dict], top_k_per_bucket: int = 3) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {
        "short_reactive": [],
        "emoji_heavy": [],
        "punctuation_heavy": [],
        "longer_expressive": [],
        "plain_typical": [],
    }

    for msg in scored_messages:
        category = categorize_message(msg["text"])
        if len(buckets[category]) < top_k_per_bucket:
            buckets[category].append(msg)

    return buckets

async def build_discriminative_profile(
    channel_id: int,
    author_id: int,
    recent_limit: int = 2000,
    feature_mode: str = "both",
    top_n_features: int = 20,
    top_k_messages: int = 10,
) -> dict | None:
    target_rows = await db.get_message_rows(channel_id, author_id, limit=recent_limit)
    if not target_rows:
        return None

    target_messages = [row["content"] for row in target_rows]
    style_stats = style.build_style_profile(target_messages)
    if style_stats is None:
        return None

    channel_rows = await db.get_message_rows(channel_id, author_id=None, limit=None)
    if not channel_rows:
        return None

    # build a local author-classification problem inside this channel
    by_author = {}
    for row in channel_rows:
        text = (row["content"] or "").strip()
        if not text:
            continue
        by_author.setdefault(row["author_id"], []).append(text)

    # keep only authors with enough messages to train something useful
    usable_authors = {
        a_id: texts
        for a_id, texts in by_author.items()
        if len(texts) >= 20
    }

    if author_id not in usable_authors:
        usable_authors[author_id] = target_messages

    texts = []
    labels = []
    label_map = {}

    for a_id, texts_for_author in usable_authors.items():
        label = f"AUTHOR_{a_id}"
        label_map[a_id] = label

        # keep data roughly balanced for this local classifier
        subset = texts_for_author[:recent_limit] if len(texts_for_author) > recent_limit else texts_for_author
        for text in subset:
            texts.append(text)
            labels.append(label)

    if len(set(labels)) < 2:
        return None

    X, word_vectorizer, char_vectorizer = build_features(feature_mode, texts)
    feature_names = build_feature_names(word_vectorizer, char_vectorizer)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(X, labels)

    target_label = label_map[author_id]
    top_word_features, top_char_features = get_top_user_features(
        model=model,
        feature_names=feature_names,
        target_label=target_label,
        top_n=top_n_features,
    )

    uniqueness_score = compute_uniqueness_score(model, target_label)

    scored_messages = []
    partial_profile = {
        "style_stats": style_stats,
        "top_word_features": top_word_features,
        "top_char_features": top_char_features,
    }

    for row in target_rows:
        text = row["content"]
        score = score_representative_message(text, partial_profile)
        scored_messages.append({
            "message_id": row["message_id"],
            "text": text,
            "created_at": row["created_at"],
            "score": score,
        })

    scored_messages.sort(key=lambda x: x["score"], reverse=True)
    representative_messages = select_diverse_messages(scored_messages, top_k=top_k_messages)
    message_buckets = build_message_buckets(scored_messages, top_k_per_bucket=3)

    return {
        "channel_id": channel_id,
        "author_id": author_id,
        "recent_limit": recent_limit,
        "feature_mode": feature_mode,
        "uniqueness_score": uniqueness_score,
        "style_stats": style_stats,
        "top_word_features": top_word_features,
        "top_char_features": top_char_features,
        "representative_messages": representative_messages,
        "message_buckets": message_buckets,
        "recent_messages": [
            {
                "message_id": row["message_id"],
                "text": row["content"],
                "created_at": row["created_at"],
            }
            for row in target_rows[:200]
        ],
    }