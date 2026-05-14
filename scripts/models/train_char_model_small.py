from emulation.train_char_model import train_model
# SMALL
if __name__ == "__main__":
    train_model(
        train_path="exports/generation_recent_5000/generation_train.txt",
        val_path="exports/generation_recent_5000/generation_val.txt",
        output_dir="exports/char_model_recent_5000_small",
        seq_len=80,
        stride=80,
        batch_size=32,
        epochs=8,
        lr=1e-3,
        embedding_dim=64,
        hidden_size=128,
        num_layers=1,
        dropout=0.1,
    )