from emulation.train_char_model import train_model


AUTHOR_LABEL = "USER_019" 


if __name__ == "__main__":
    train_model(
        train_path=f"exports/generation_{AUTHOR_LABEL.lower()}/generation_train.txt",
        val_path=f"exports/generation_{AUTHOR_LABEL.lower()}/generation_val.txt",
        output_dir=f"exports/char_model_{AUTHOR_LABEL.lower()}",
        seq_len=160,
        stride=100,
        batch_size=24,
        epochs=18,
        lr=8e-4,
        embedding_dim=192,
        hidden_size=384,
        num_layers=2,
        dropout=0.2,
    )