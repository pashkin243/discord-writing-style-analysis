from emulation.train_char_model import train_model
# LARGE
if __name__ == "__main__":
    train_model(
        train_path="exports/generation_recent_5000/generation_train.txt",
        val_path="exports/generation_recent_5000/generation_val.txt",
        output_dir="exports/char_model_recent_5000_large",
        seq_len=160,
        stride=100,
        batch_size=24,
        epochs=15,
        lr=8e-4,
        embedding_dim=192,
        hidden_size=384,
        num_layers=2,
        dropout=0.2,
    )