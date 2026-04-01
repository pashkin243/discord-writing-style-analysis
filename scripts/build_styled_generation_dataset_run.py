from emulation.build_styled_generation_dataset import build_generation_dataset


if __name__ == "__main__":
    build_generation_dataset(
        input_path="exports/messages_recent_5000.jsonl",
        output_dir="exports/generation_recent_5000",
    )