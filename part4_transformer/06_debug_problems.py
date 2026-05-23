"""
Transformer debugging snippets collected as safe, runnable helpers.

Intentionally broken lesson snippets are represented as comments in the source
material; this file exposes corrected utilities without top-level pseudo-code.
"""

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


def create_causal_mask(seq_len, device=None):
    device = device or "cpu"
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_padding_mask(seq_lengths, max_len):
    batch_size = len(seq_lengths)
    device = seq_lengths.device
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len)
    mask = mask < seq_lengths.unsqueeze(1)
    return mask.unsqueeze(1).unsqueeze(2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


def multi_head_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5),
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def beam_search(model, start_token, max_len, beam_size=5):
    device = next(model.parameters()).device
    beams = torch.full((beam_size, 1), start_token, dtype=torch.long, device=device)
    beam_scores = torch.zeros(beam_size, device=device)
    beam_scores[1:] = -float("inf")

    for _ in range(max_len):
        logits = model(beams)[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        candidate_scores = (beam_scores.unsqueeze(1) + log_probs).reshape(-1)
        top_scores, top_indices = torch.topk(candidate_scores, beam_size)
        beam_indices = top_indices // logits.size(-1)
        token_indices = top_indices % logits.size(-1)
        beams = torch.cat([beams[beam_indices], token_indices.unsqueeze(1)], dim=1)
        beam_scores = top_scores

    return beams[beam_scores.argmax()]


def check_nan_inf(model):
    issues = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append((name, "grad_nan"))
            if torch.isinf(param.grad).any():
                issues.append((name, "grad_inf"))
        if torch.isnan(param).any():
            issues.append((name, "param_nan"))
        if torch.isinf(param).any():
            issues.append((name, "param_inf"))
    return issues


def plot_attention(attn_weights, tokens):
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights.detach().cpu().numpy(), xticklabels=tokens, yticklabels=tokens, cmap="Blues")
    plt.xlabel("Key")
    plt.ylabel("Query")
    return plt.gcf()


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
    plt.figure(figsize=(10, 4))
    plt.plot(ave_grads)
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.ylabel("Average gradient")
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    Q = torch.randn(2, 4, 8, 16)
    K = torch.randn(2, 4, 8, 16)
    V = torch.randn(2, 4, 8, 16)
    out, attn = scaled_dot_product_attention(Q, K, V, create_causal_mask(8))
    assert out.shape == Q.shape
    assert attn.shape == (2, 4, 8, 8)
    print("Transformer debug helpers smoke test passed.")
