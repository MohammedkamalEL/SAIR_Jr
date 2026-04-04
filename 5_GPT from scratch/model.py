"""
model.py  —  GPT-2 from Scratch
SAIR Module 5: GPT from Scratch | Lecture 3

All classes a student needs to understand and implement GPT-2 Small (124M params).
Run this file directly to verify everything works: `uv run python model.py`
"""
import torch
import torch.nn as nn


# ── Configuration ─────────────────────────────────────────────────────────────
GPT_CONFIG_124 = {
    "vocab_size":     50257,   # BPE vocabulary (GPT-2)
    "context_length":  1024,   # Maximum sequence length
    "emb_dim":          768,   # Hidden dimension
    "n_heads":           12,   # Attention heads
    "n_layers":          12,   # Transformer blocks
    "dropout":          0.1,   # Dropout rate
    "qkv_bias":        False,  # No bias on QKV projections
}


# ── Building Block 1: Layer Normalization ─────────────────────────────────────
class LayerNorm(nn.Module):
    """
    Normalizes activations across the feature dimension (not the batch).
    Keeps activations well-scaled through all 12 transformer blocks.
    γ and β are learned — the network can undo normalization if needed.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))    # scale — learned
        self.beta  = nn.Parameter(torch.zeros(emb_dim))   # shift — learned
        self.eps   = 1e-5

    def forward(self, x):
        mean  = x.mean(dim=-1, keepdim=True)
        var   = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# ── Building Block 2: GELU Activation ────────────────────────────────────────
class GELU(nn.Module):
    """
    Smooth activation used in GPT-2. Unlike ReLU, it allows small negative
    outputs near zero — preventing dead neurons and improving gradient flow.
    This is the tanh approximation used in the original GPT-2 implementation.
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x.pow(3))
        ))


# ── Building Block 3: FeedForward Network ────────────────────────────────────
class FeedForward(nn.Module):
    """
    Per-token transformation applied after attention.
    Expands dimension by 4× to create a richer working space, then projects back.
    Stores factual knowledge; attention decides what to look at, FFN what to do.

    Shape: (B, T, d) → (B, T, d)   unchanged — runs independently per token
    """
    def __init__(self, config):
        super().__init__()
        d = config["emb_dim"]
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),          # expand:  768 → 3072
            GELU(),                        # smooth nonlinearity
            nn.Linear(4 * d, d),          # project: 3072 → 768
            nn.Dropout(config["dropout"]),
        )

    def forward(self, x):
        return self.net(x)


# ── Building Block 4: Multi-Head Causal Self-Attention ───────────────────────
class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention (from Notebook 2).
    Each token attends to all past tokens but NOT future ones (causal mask).
    Multiple heads allow attending to different aspects of the context in parallel.

    Shape: (B, T, d) → (B, T, d)
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_q  = nn.Linear(d_in, d_out, bias=bias)
        self.W_k  = nn.Linear(d_in, d_out, bias=bias)
        self.W_v  = nn.Linear(d_in, d_out, bias=bias)
        self.proj = nn.Linear(d_out, d_out)            # output projection
        self.drop = nn.Dropout(dropout)

        # Causal mask: upper-triangular → future positions get -inf → softmax → 0
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) * self.head_dim ** -0.5
        scores  = scores.masked_fill(self.mask[:T, :T].bool(), float("-inf"))
        weights = self.drop(torch.softmax(scores, dim=-1))

        out = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.proj(out)


# ── Building Block 5: Transformer Block ──────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    The core repeating unit of GPT-2. Stacked 12 times.

    Pre-norm style (GPT-2): normalize BEFORE each sub-layer, then add residual.
        x = x + Attention(LayerNorm(x))    ← residual 1
        x = x + FFN(LayerNorm(x))          ← residual 2

    Shape preserved: (B, T, d) → (B, T, d)
    """
    def __init__(self, config):
        super().__init__()
        d = config["emb_dim"]
        self.norm1 = LayerNorm(d)
        self.attn  = MultiHeadAttention(
            d_in=d, d_out=d,
            context_length=config["context_length"],
            dropout=config["dropout"],
            num_heads=config["n_heads"],
            bias=config["qkv_bias"],
        )
        self.norm2 = LayerNorm(d)
        self.ffn   = FeedForward(config)
        self.drop  = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))   # attention sub-layer
        x = x + self.drop(self.ffn(self.norm2(x)))    # FFN sub-layer
        return x


# ── Full Model: GPT-2 Small ───────────────────────────────────────────────────
class GPTModel(nn.Module):
    """
    GPT-2 Small (124M parameters).

    Forward pass:
        token IDs (B, T)
            ↓  tok_emb + pos_emb
        input embeddings (B, T, 768)
            ↓  12 × TransformerBlock
        contextualized embeddings (B, T, 768)
            ↓  LayerNorm + Linear
        logits (B, T, 50257)
    """
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"],     config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop    = nn.Dropout(config["dropout"])
        self.blocks  = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.norm    = LayerNorm(config["emb_dim"])
        self.head    = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

        # Weight tying: output head shares the token embedding matrix.
        # Saves 50257 × 768 = 38.6M params. Standard in GPT-2 and most modern LLMs.
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):                                      # idx: (B, T)
        B, T = idx.shape
        tok  = self.tok_emb(idx)                                 # (B, T, C)
        pos  = self.pos_emb(torch.arange(T, device=idx.device)) # (T, C)
        x    = self.drop(tok + pos)                              # (B, T, C)
        x    = self.blocks(x)                                    # (B, T, C)
        x    = self.norm(x)                                      # (B, T, C)
        return self.head(x)                                      # (B, T, vocab_size)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    model  = GPTModel(GPT_CONFIG_124)
    params = sum(p.numel() for p in model.parameters())

    ids    = torch.randint(0, GPT_CONFIG_124["vocab_size"], (2, 10))
    logits = model(ids)

    print(f"Input  shape : {tuple(ids.shape)}")
    print(f"Output shape : {tuple(logits.shape)}")
    print(f"Total params : {params:,}  ({params/1e6:.1f}M)")
    assert logits.shape == (2, 10, 50257), "Shape mismatch!"
    assert abs(params/1e6 - 124.4) < 1.0,  "Unexpected parameter count!"
    print("All checks passed ✓")
