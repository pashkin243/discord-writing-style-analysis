import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int, stoi: dict[str, int], stride: int = 40):
        self.text = text
        self.seq_len = seq_len
        self.stoi = stoi
        self.stride = stride
        self.start_positions = list(range(0, max(1, len(text) - seq_len - 1), stride))

    def __len__(self) -> int:
        return len(self.start_positions)

    def __getitem__(self, idx: int):
        start = self.start_positions[idx]
        chunk = self.text[start : start + self.seq_len + 1]
        x = torch.tensor([self.stoi[c] for c in chunk[:-1]], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in chunk[1:]], dtype=torch.long)
        return x, y


class CharLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def evaluate(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity


def train_model(
    train_path: str,
    val_path: str,
    output_dir: str,
    seq_len: int = 120,
    stride: int = 40,
    batch_size: int = 64,
    epochs: int = 8,
    lr: float = 1e-3,
    embedding_dim: int = 128,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_text = read_text(Path(train_path))
    val_text = read_text(Path(val_path))

    if not train_text.strip():
        raise RuntimeError("Training text is empty.")
    if not val_text.strip():
        raise RuntimeError("Validation text is empty.")

    full_text_for_vocab = train_text + val_text
    chars, stoi, itos = build_vocab(full_text_for_vocab)

    train_dataset = CharDataset(train_text, seq_len=seq_len, stoi=stoi, stride=stride)
    val_dataset = CharDataset(val_text, seq_len=seq_len, stoi=stoi, stride=stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = CharLSTM(
        vocab_size=len(chars),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    history = []

    print(f"Device: {DEVICE}")
    print(f"Vocab size: {len(chars)}")
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()
        total_loss = 0.0
        total_tokens = 0

        batch_count = len(train_loader)
        print(f"\n--- Epoch {epoch}/{epochs} started ---")

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            batch_start = time.time()

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            if batch_idx == 1 or batch_idx % 200 == 0 or batch_idx == batch_count:
                elapsed = time.time() - batch_start
                running_loss = total_loss / max(1, total_tokens)
                running_ppl = math.exp(running_loss) if running_loss < 20 else float("inf")
                print(
                    f"Epoch {epoch} | batch {batch_idx}/{batch_count} | "
                    f"batch_time={elapsed:.2f}s | running_loss={running_loss:.4f} | "
                    f"running_ppl={running_ppl:.2f}",
                    flush=True,
                )

        train_loss = total_loss / max(1, total_tokens)
        train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")

        val_start = time.time()
        val_loss, val_ppl = evaluate(model, val_loader, criterion)
        val_elapsed = time.time() - val_start

        epoch_elapsed = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_perplexity": train_ppl,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "epoch_time_seconds": epoch_elapsed,
                "val_time_seconds": val_elapsed,
            }
        )

        print(
            f"Epoch {epoch}/{epochs} finished | "
            f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} | "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} | "
            f"val_time={val_elapsed:.2f}s | epoch_time={epoch_elapsed:.2f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "chars": chars,
                "config": {
                    "seq_len": seq_len,
                    "stride": stride,
                    "embedding_dim": embedding_dim,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "vocab_size": len(chars),
                },
            }

            torch.save(checkpoint, output_dir / "char_lstm_best.pt")
            print("Saved new best model checkpoint.", flush=True)

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "device": DEVICE,
                "vocab_size": len(chars),
                "train_sequences": len(train_dataset),
                "val_sequences": len(val_dataset),
                "best_val_loss": best_val_loss,
                "epochs": epochs,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "stride": stride,
                "learning_rate": lr,
            },
            f,
            indent=2,
        )

    print(f"\nSaved best model to {output_dir / 'char_lstm_best.pt'}")
    print(f"Saved training history to {output_dir / 'training_history.json'}")
    print(f"Saved training summary to {output_dir / 'training_summary.json'}")

# Kasutatud AI