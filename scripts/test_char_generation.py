from emulation.generate_char_model import load_model, generate_text, clean_generated_output


MODEL_PATH = "exports/char_model_recent_5000_large/char_lstm_best.pt"
USER_TOKEN = "<USER_000>"


def generate_nonempty(model, stoi, itos, prompt, max_tries=10):
    for _ in range(max_tries):
        raw = generate_text(
            model,
            stoi,
            itos,
            prompt,
            max_new_chars=160,
            temperature=0.75,
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

# Kasutatud AI