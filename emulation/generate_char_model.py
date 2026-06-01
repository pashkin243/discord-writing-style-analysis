import torch
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CharLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)

    model = CharLSTM(
        vocab_size=checkpoint["config"]["vocab_size"],
        embedding_dim=checkpoint["config"]["embedding_dim"],
        hidden_size=checkpoint["config"]["hidden_size"],
        num_layers=checkpoint["config"]["num_layers"],
        dropout=checkpoint["config"]["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["stoi"], checkpoint["itos"], checkpoint["config"]


def generate_text(model, stoi, itos, prompt, max_new_chars=160, temperature=0.5):
    hidden = None

    input_ids = [stoi[c] for c in prompt if c in stoi]
    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits, hidden = model(x, hidden)

    generated = prompt
    last_idx = x[0, -1].view(1, 1)

    for _ in range(max_new_chars):
        logits, hidden = model(last_idx, hidden)
        logits = logits[0, -1] / temperature

        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()

        next_char = itos[next_idx]
        generated += next_char

        last_idx = torch.tensor([[next_idx]], device=DEVICE)

        if next_char == "\n":
            break

    return generated


def clean_generated_output(text, prompt):
    if text.startswith(prompt):
        text = text[len(prompt):]

    return text.strip()

# Kasutatud AI