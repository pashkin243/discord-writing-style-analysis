from emulation.generate_char_model import load_model, generate_text, clean_generated_output

AUTHOR_LABEL = "USER_003"
MODEL_PATH = f"exports/char_model_{AUTHOR_LABEL.lower()}/char_lstm_best.pt"
USER_TOKEN = f"<{AUTHOR_LABEL}>"


def generate_nonempty(model, stoi, itos, prompt, max_tries=10):
    for _ in range(max_tries):
        raw = generate_text(
            model=model,
            stoi=stoi,
            itos=itos,
            prompt=prompt,
            max_new_chars=160,
            temperature=0.5,
        )
        cleaned = clean_generated_output(raw, prompt)
        if cleaned and len(cleaned) >= 4:
            return cleaned
    return "[generation failed]"


if __name__ == "__main__":
    model, stoi, itos, config = load_model(MODEL_PATH)

    print("=== GENERATED ===")
    for i in range(10):
        print(f"\n[{i+1}] {generate_nonempty(model, stoi, itos, USER_TOKEN + ' ')}")