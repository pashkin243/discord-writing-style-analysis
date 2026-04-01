from emulation.build_single_user_generation_dataset import build_single_user_generation_dataset


AUTHOR_LABEL = "USER_000"  # change this


if __name__ == "__main__":
    build_single_user_generation_dataset(
        input_path="exports/messages_recent_5000.jsonl",
        output_dir=f"exports/generation_{AUTHOR_LABEL.lower()}",
        author_label=AUTHOR_LABEL,
        min_messages=500,
        max_messages=5000,
    )