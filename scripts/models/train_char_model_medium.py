from emulation.train_char_model import train_model
# MEDIUM
if __name__ == "__main__":
    train_model(
        train_path="exports/generation_recent_5000/generation_train.txt",
        val_path="exports/generation_recent_5000/generation_val.txt",
        output_dir="exports/char_model_recent_5000_medium",
        seq_len=120,
        stride=80,
        batch_size=32,
        epochs=12,
        lr=1e-3,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    )