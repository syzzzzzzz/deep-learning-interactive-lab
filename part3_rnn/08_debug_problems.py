try:
    """
    RNN debugging snippets collected as safe, runnable helpers.

    The original lesson contains intentionally broken examples. This module keeps
    the corrected patterns executable without running undefined pseudo-code at the
    top level.
    """

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


    def clip_training_step(model, optimizer, loss, max_norm=1.0):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()


    def init_lstm_hidden(model, batch_size, device=None):
        device = device or next(model.parameters()).device
        directions = 2 if getattr(model, "bidirectional", False) else 1
        num_layers = getattr(model, "num_layers", 1)
        hidden_size = getattr(model, "hidden_size", 1)
        h = torch.zeros(num_layers * directions, batch_size, hidden_size, device=device)
        c = torch.zeros(num_layers * directions, batch_size, hidden_size, device=device)
        return h, c


    def pad_and_pack(sequences, embedding, lstm):
        padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(s) for s in sequences], device=padded.device)
        embedded = embedding(padded)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden


    def masked_cross_entropy(logits, targets, pad_index=0):
        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        mask = targets.reshape(-1) != pad_index
        denom = mask.sum().clamp_min(1)
        return (flat_loss * mask).sum() / denom


    def get_teacher_forcing_ratio(epoch, k=10):
        return k / (k + np.exp(epoch / k))


    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, x):
            embedded = self.embedding(x)
            _, (h_n, _) = self.lstm(embedded)
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            return self.fc(h)


    class ProperDropoutLSTM(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_layers=2, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(
                embed_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            output, _ = self.lstm(self.embedding(x))
            return self.fc(self.dropout(output[:, -1, :]))


    class SingleLayerLSTMWithDropout(nn.Module):
        def __init__(self, embed_size, hidden_size, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, h=None):
            return self.lstm(self.dropout(x), h)


    def generate_with_control(model, start_token, max_len=50, temperature=1.0, top_k=0, top_p=0.0):
        model.eval()
        x = torch.tensor([[start_token]], device=next(model.parameters()).device)
        h = None
        generated = []

        with torch.no_grad():
            for _ in range(max_len):
                logits, h = model(x, h)
                logits = logits[0, -1, :] / max(temperature, 1e-8)

                if top_k > 0:
                    top_vals, _ = logits.topk(min(top_k, logits.numel()))
                    logits[logits < top_vals[-1]] = -float("inf")

                if top_p > 0:
                    sorted_logits, sorted_idx = logits.sort(descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs > top_p
                    remove_mask[1:] = remove_mask[:-1].clone()
                    remove_mask[0] = False
                    logits[sorted_idx[remove_mask]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token.item())
                x = next_token.view(1, 1)

        return generated


    def check_rnn_gradient_flow(model):
        rows = []
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            grad_norm = p.grad.norm().item()
            grad_mean = p.grad.abs().mean().item()
            grad_max = p.grad.abs().max().item()
            rows.append((name, grad_norm, grad_mean, grad_max))
        return rows


    def check_hidden_state(hidden):
        h, c = hidden
        return {
            "h_norm": h.norm().item(),
            "c_norm": c.norm().item(),
            "h_has_nan": torch.isnan(h).any().item(),
            "c_has_inf": torch.isinf(c).any().item(),
        }


    if __name__ == "__main__":
        model = BiLSTMClassifier(vocab_size=12, embed_size=4, hidden_size=6, num_classes=3)
        x = torch.randint(0, 12, (2, 5))
        y = model(x)
        assert y.shape == (2, 3)
        print("RNN debug helpers smoke test passed.")
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part3_rnn/08_debug_problems.py", e)
