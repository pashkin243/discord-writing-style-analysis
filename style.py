#stiilianalüüs toimub siin failis.
#kasutatakse ka regexit.
from collections import Counter
import re

def build_style_profile(messages: list[str]) -> dict | None:
    if not messages:
        return None
    
    total_messages = len(messages)
    total_chars = sum(len(m) for m in messages)
    total_words = sum(len(m.split()) for m in messages)
    exclamations = sum(m.count("!") for m in messages)
    questions = sum(m.count("?") for m in messages)
    periods = sum(m.count(".") for m in messages)
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

    # tokeniseerimine
    words = []
    for msg in messages:
        tokens = re.findall(r"\b\w+\b", msg.lower())
        words.extend(tokens)
    common_words = Counter(words).most_common(10)

    return {
        "messages": total_messages,
        "avg_length": avg_length,
        "avg_words": avg_words,
        "exclamations_per_msg": exclamations / total_messages,
        "questions_per_msg": questions / total_messages,
        "periods_per_msg": periods / total_messages,
        "uppercase_ratio": uppercase_ratio,
        "common_words": common_words,
    }