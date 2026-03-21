#stiilianalüüs toimub siin failis.
#kasutatakse ka regexit.
from collections import Counter
import re

# korduvad kirjavahemärgid
def get_punctuation_runs(messages: list[str], symbol: str, min_length: int = 1) -> list[int]:
    runs = []
    pattern = re.escape(symbol) + r"+"

    for msg in messages:
        matches = re.findall(pattern, msg)
        for match in matches:
            if len(match) >= min_length:
                runs.append(len(match))
    return runs

# discordi custom emojid
def extract_custom_emojis(text: str) -> list[str]:
    return re.findall(r"<a?:\w+:\d+>", text)

# unicode emojid
def extract_unicode_emojis(text: str) -> list[str]:
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF" 
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.findall(text)


# kõik emojid
def extract_all_emojis(text: str) -> list[str]:
    custom_emojis = extract_custom_emojis(text)
    unicode_emojis = extract_unicode_emojis(text)
    return custom_emojis + unicode_emojis

def build_style_profile(messages: list[str]) -> dict | None:
    if not messages:
        return None
    
    total_messages = len(messages)
    total_chars = sum(len(m) for m in messages)
    total_words = sum(len(m.split()) for m in messages)

    exclamations = sum(m.count("!") for m in messages)
    questions = sum(m.count("?") for m in messages)
    periods = sum(m.count(".") for m in messages)
    commas = sum(m.count(",") for m in messages)
    newlines = sum(m.count("\n") for m in messages)

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
    comma_message_ratio = sum(1 for m in messages if "," in m) / total_messages
    avg_newlines_per_msg = newlines / total_messages

    exclamation_runs = get_punctuation_runs(messages, "!", min_length=1)
    question_runs = get_punctuation_runs(messages, "?", min_length=1)
    dot_runs = get_punctuation_runs(messages, ".", min_length=2)
    exclamation_run_rate = len(exclamation_runs) / total_messages
    question_run_rate = len(question_runs) / total_messages
    dot_run_rate = len(dot_runs) / total_messages
    avg_exclamation_run_length = (
        sum(exclamation_runs) / len(exclamation_runs) if exclamation_runs else 0
    )
    avg_question_run_length = (
        sum(question_runs) / len(question_runs) if question_runs else 0
    )
    avg_dot_run_length = (
        sum(dot_runs) / len(dot_runs) if dot_runs else 0
    )
    max_exclamation_run_length = max(exclamation_runs) if exclamation_runs else 0
    max_question_run_length = max(question_runs) if question_runs else 0
    max_dot_run_length = max(dot_runs) if dot_runs else 0

    # tokeniseerimine
    words = []
    for msg in messages:
        tokens = re.findall(r"\b\w+\b", msg.lower())
        words.extend(tokens)
    common_words = Counter(words).most_common(10)

    # bigrammid
    bigrams = []
    for msg in messages:
        tokens = re.findall(r"\b\w+\b", msg.lower())
        for i in range(len(tokens) - 1):
            bigrams.append((tokens[i], tokens[i + 1]))
    common_bigrams = Counter(bigrams).most_common(10)

    # emojid
    all_emojis = []
    emoji_message_count = 0
    for msg in messages:
        emojis = extract_all_emojis(msg)
        if emojis:
            emoji_message_count += 1
            all_emojis.extend(emojis)
    common_emojis = Counter(all_emojis).most_common(10)
    emojis_per_msg = len(all_emojis) / total_messages
    emoji_message_ratio = emoji_message_count / total_messages

    return {
        "messages": total_messages,
        "avg_length": avg_length,
        "avg_words": avg_words,
        "exclamations_per_msg": exclamations / total_messages,
        "questions_per_msg": questions / total_messages,
        "periods_per_msg": periods / total_messages,
        "commas_per_msg": commas / total_messages,
        "comma_message_ratio": comma_message_ratio,
        "avg_newlines_per_msg": avg_newlines_per_msg,
        "uppercase_ratio": uppercase_ratio,
        "exclamation_run_rate": exclamation_run_rate,
        "question_run_rate": question_run_rate,
        "dot_run_rate": dot_run_rate,
        "avg_exclamation_run_length": avg_exclamation_run_length,
        "avg_question_run_length": avg_question_run_length,
        "avg_dot_run_length": avg_dot_run_length,
        "max_exclamation_run_length": max_exclamation_run_length,
        "max_question_run_length": max_question_run_length,
        "max_dot_run_length": max_dot_run_length,
        "common_words": common_words,
        "common_bigrams": common_bigrams,
        "common_emojis": common_emojis,
        "emojis_per_msg": emojis_per_msg,
        "emoji_message_ratio": emoji_message_ratio,
    }