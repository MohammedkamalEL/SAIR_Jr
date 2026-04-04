"""
gpt_viz.py  —  Visualization helpers for Lecture 3: GPT from Scratch
All matplotlib-heavy code lives here so the notebook stays focused on GPT concepts.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── consistent style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
})
COLORS = {
    "blue":   "#3498db",
    "orange": "#e67e22",
    "green":  "#2ecc71",
    "red":    "#e74c3c",
    "purple": "#8e44ad",
    "gray":   "#95a5a6",
    "yellow": "#f1c40f",
}


# ────────────────────────────────────────────────────────────────────────────
def draw_architecture(config):
    """Draw the full GPT-2 architecture as a vertical stack."""
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, 10); ax.set_ylim(0, 23); ax.axis("off")
    fig.patch.set_facecolor("#fafafa")

    def box(y, label, sub="", color="#3498db", h=0.85, alpha=0.9):
        rect = FancyBboxPatch((1.2, y), 7.6, h,
                              boxstyle="round,pad=0.12",
                              facecolor=color, edgecolor="white", lw=2, alpha=alpha)
        ax.add_patch(rect)
        ax.text(5, y + h/2 + (0.13 if sub else 0), label,
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        if sub:
            ax.text(5, y + h/2 - 0.16, sub,
                    ha="center", va="center", fontsize=8.5, color="#dce8ff")

    def arrow(y_from, y_to, shape=""):
        ax.annotate("", xy=(5, y_to), xytext=(5, y_from),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.8))
        if shape:
            ax.text(6.5, (y_from + y_to) / 2, shape,
                    fontsize=8, color="#444", va="center",
                    bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.2"))

    # ── bottom to top ────────────────────────────────────────────────────────
    box(0.4,  "Input Token IDs",          "(B, T)  —  integers",          "#7f8c8d")
    arrow(1.25, 1.65, f"(B,T)")
    box(1.65, "Token Embedding",          f"(B,T) → (B,T,{config['emb_dim']})",  "#e67e22")
    box(2.60, "Positional Embedding",     f"(T)   → (T,{config['emb_dim']})",    "#e67e22")
    ax.text(5, 3.55, "+", ha="center", va="center", fontsize=22, color="#e67e22", fontweight="bold")
    arrow(3.65, 4.0, f"(B,T,{config['emb_dim']})")

    # Transformer blocks bracket
    n     = config["n_layers"]
    bh    = 0.72
    y0    = 4.2
    blues = ["#2980b9", "#2471a3", "#1f618d", "#1a5276"]
    for i in range(n):
        c = blues[i % len(blues)]
        rect = FancyBboxPatch((1.2, y0 + i*bh), 7.6, bh - 0.06,
                              boxstyle="round,pad=0.06",
                              facecolor=c, edgecolor="white", lw=1.2, alpha=0.88)
        ax.add_patch(rect)
        ax.text(5, y0 + i*bh + (bh-0.06)/2,
                f"Transformer Block {i+1:2d}   |   LN → MHA + skip   |   LN → FFN + skip",
                ha="center", va="center", fontsize=7.5, color="white")

    bracket_y = y0 + n * bh
    ax.annotate("", xy=(0.8, bracket_y), xytext=(0.8, y0),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
    ax.text(0.35, (y0 + bracket_y) / 2, f"×{n}",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#333")

    arrow(bracket_y, bracket_y + 0.4, f"(B,T,{config['emb_dim']})")
    y2 = bracket_y + 0.4
    box(y2,      "Final LayerNorm",       f"(B,T,{config['emb_dim']})",          "#27ae60")
    arrow(y2+0.9, y2+1.3, f"(B,T,{config['emb_dim']})")
    y3 = y2 + 1.3
    box(y3,      "Output Linear (LM head)", f"(B,T,{config['emb_dim']}) → (B,T,{config['vocab_size']})", "#8e44ad")

    params_approx = 124
    ax.text(5, y3 + 1.15,
            f"GPT-2 Small  —  ~{params_approx}M Parameters",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#222")
    plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────
def plot_embedding_heatmap(vocab=12, dim=10, seed=42):
    """Show the embedding matrix and what a lookup produces."""
    torch.manual_seed(seed)
    E = nn.Embedding(vocab, dim).weight.data.numpy()
    selected = [2, 6, 9]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    kw = dict(cmap="RdBu", aspect="auto", vmin=-1.5, vmax=1.5)

    im = axes[0].imshow(E, **kw)
    axes[0].set_title("Embedding Matrix  E  ∈  ℝ^(vocab × dim)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Embedding dimension"); axes[0].set_ylabel("Token ID")
    plt.colorbar(im, ax=axes[0])
    for t in selected:
        axes[0].add_patch(plt.Rectangle((-0.5, t-0.5), dim, 1,
                                        fill=False, edgecolor="gold", lw=3))
    axes[0].set_yticks(range(vocab))
    axes[0].set_yticklabels([f"tok {i}" for i in range(vocab)], fontsize=8)

    im2 = axes[1].imshow(E[selected], **kw)
    axes[1].set_title(f"Looked-up rows for token IDs {selected}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Embedding dimension")
    axes[1].set_yticks(range(len(selected)))
    axes[1].set_yticklabels([f"tok {t}" for t in selected])
    plt.colorbar(im2, ax=axes[1])

    plt.suptitle("Token Embedding: Integer → Dense Vector (lookup table)", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────
def plot_positional_heatmap(max_pos=32, dim=64, seed=7):
    """2-D heatmap showing how positional embeddings vary over positions."""
    torch.manual_seed(seed)
    P = nn.Embedding(max_pos, dim).weight.data.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    im = axes[0].imshow(P, cmap="RdBu", aspect="auto", vmin=-1.5, vmax=1.5)
    axes[0].set_title("Positional Embedding Matrix  P  ∈  ℝ^(max_pos × dim)",
                       fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Embedding dimension"); axes[0].set_ylabel("Position index")
    plt.colorbar(im, ax=axes[0])

    # Show a few position vectors as line plots
    for i, (pos, color) in enumerate([(0, COLORS["blue"]), (8, COLORS["orange"]),
                                       (16, COLORS["green"]), (31, COLORS["red"])]):
        axes[1].plot(P[pos, :32], color=color, lw=1.8, label=f"position {pos}")
    axes[1].set_title("First 32 dims of selected position vectors", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Dimension index"); axes[1].set_ylabel("Value"); axes[1].legend()

    plt.suptitle("Positional Embeddings: Each Position Gets a Unique Signature",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────
def plot_input_pipeline(seq_len=8, dim=12, seed=42):
    """Show token_emb + pos_emb = input_emb as three side-by-side heatmaps."""
    torch.manual_seed(seed)
    tok_layer = nn.Embedding(100, dim)
    pos_layer = nn.Embedding(seq_len, dim)
    ids  = torch.randint(0, 100, (seq_len,))
    T    = tok_layer(ids).detach().numpy()
    P    = pos_layer(torch.arange(seq_len)).detach().numpy()
    X    = T + P

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    kw = dict(cmap="RdBu", aspect="auto", vmin=-2, vmax=2)
    titles = ["Token Embedding  (T)", "Positional Embedding  (P)",
              "Input Embedding  X = T + P"]
    for ax, mat, title in zip(axes, [T, P, X], titles):
        im = ax.imshow(mat, **kw)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Dim"); ax.set_ylabel("Token position")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # "+" annotation between panels
    for x_pos in [0.365, 0.69]:
        fig.text(x_pos, 0.52, "+", ha="center", va="center",
                 fontsize=24, fontweight="bold", color="#333",
                 transform=fig.transFigure)

    plt.suptitle("Input Embedding Pipeline: What the Transformer Actually Receives",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────
def plot_layernorm_effect(emb_dim=768, n_samples=64, seed=0):
    """Before/after LayerNorm histogram + per-sample mean scatter."""
    torch.manual_seed(seed)

    class LayerNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(d))
            self.beta  = nn.Parameter(torch.zeros(d))
        def forward(self, x):
            m = x.mean(-1, keepdim=True); v = x.var(-1, keepdim=True, unbiased=False)
            return self.gamma * (x - m) / (v + 1e-5).sqrt() + self.beta

    x_raw   = torch.randn(n_samples, emb_dim) * 4 + 7
    x_norm  = LayerNorm(emb_dim)(x_raw).detach()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(x_raw.numpy().flatten(),  bins=80, color=COLORS["red"],   alpha=0.8, edgecolor="white")
    axes[0].axvline(x_raw.mean().item(), color="black", lw=2, linestyle="--",
                    label=f"mean = {x_raw.mean():.1f}")
    axes[0].set_title("Before LayerNorm", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Activation value"); axes[0].set_ylabel("Count"); axes[0].legend()

    axes[1].hist(x_norm.numpy().flatten(), bins=80, color=COLORS["green"], alpha=0.8, edgecolor="white")
    axes[1].axvline(x_norm.mean().item(), color="black", lw=2, linestyle="--",
                    label=f"mean ≈ {x_norm.mean():.3f}")
    axes[1].set_title("After LayerNorm", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Activation value"); axes[1].legend()

    means_before = x_raw.mean(dim=-1).detach().numpy()
    means_after  = x_norm.mean(dim=-1).detach().numpy()
    axes[2].scatter(range(n_samples), means_before, label="Before", color=COLORS["red"],  s=40, alpha=0.7)
    axes[2].scatter(range(n_samples), means_after,  label="After",  color=COLORS["green"], s=40, alpha=0.7)
    axes[2].axhline(0, color="black", lw=1.2, linestyle="--")
    axes[2].set_title("Per-sample mean — before vs after", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Sample index"); axes[2].set_ylabel("Mean"); axes[2].legend()

    plt.suptitle("LayerNorm: Taming Wild Activations", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
    print(f"Before  — mean: {x_raw.mean():.2f},  std: {x_raw.std():.2f}")
    print(f"After   — mean: {x_norm.mean():.4f}, std: {x_norm.std():.4f}")


# ────────────────────────────────────────────────────────────────────────────
def plot_gelu_comparison():
    """Three-panel: custom vs PyTorch GELU, GELU vs ReLU, difference."""
    class GELU(nn.Module):
        def forward(self, x):
            return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x+0.044715*x**3)))

    x = torch.linspace(-4, 4, 300)
    gc = GELU()(x).detach()
    gp = nn.GELU()(x).detach()
    r  = nn.ReLU()(x).detach()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(x, gc, lw=3, color=COLORS["blue"],   label="Custom GELU")
    axes[0].plot(x, gp, lw=2, color=COLORS["red"], linestyle="--", label="PyTorch GELU")
    axes[0].set_title("Custom vs PyTorch GELU", fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
    axes[0].axhline(0, color="gray", lw=0.8); axes[0].axvline(0, color="gray", lw=0.8)

    axes[1].plot(x, gc, lw=3, color=COLORS["blue"],   label="GELU (smooth)")
    axes[1].plot(x, r,  lw=3, color=COLORS["orange"], label="ReLU (hard cutoff)")
    axes[1].set_title("GELU vs ReLU", fontsize=12, fontweight="bold")
    axes[1].legend(); axes[1].set_xlabel("x")
    axes[1].axhline(0, color="gray", lw=0.8); axes[1].axvline(0, color="gray", lw=0.8)

    diff = gc - r
    axes[2].fill_between(x, diff, alpha=0.35, color=COLORS["purple"])
    axes[2].plot(x, diff, lw=2, color=COLORS["purple"])
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_title("Difference (GELU − ReLU)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("GELU − ReLU")
    axes[2].text(-3.8, -0.18, "GELU allows small\nnegative activations",
                 fontsize=9, color=COLORS["purple"])

    plt.suptitle("GELU — Smooth Gradients, No Dead Neurons", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()
    print(f"Max difference between custom and PyTorch GELU: {(gc-gp).abs().max().item():.2e}  ✓")


# ────────────────────────────────────────────────────────────────────────────
def plot_ffn_expansion(config):
    """Annotated diagram showing 768 → 3072 → 768 expansion."""
    d = config["emb_dim"];  h = 4 * d;  n = config["n_layers"]
    params_per = 2 * d * h
    params_all = params_per * n

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(-0.5, 5); ax.set_ylim(0, 3.5); ax.axis("off")

    def fbox(x, label, sub, color, w=0.9):
        rect = FancyBboxPatch((x - w/2, 0.9), w, 1.6,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor="white", lw=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, 1.7 + 0.12, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(x, 1.7 - 0.22, sub,   ha="center", va="center",
                fontsize=9, color="#eee")

    fbox(0.3, "Input",   f"d = {d}",   COLORS["blue"])
    ax.annotate("", xy=(1.25, 1.7), xytext=(0.75, 1.7),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2))
    ax.text(1.0, 2.1, f"W₁\n({d}→{h})", ha="center", fontsize=8.5, color="#555")

    fbox(1.9, "Hidden",  f"d = {h}\n(4×)", COLORS["orange"])
    ax.text(1.9, 0.65, "GELU activation", ha="center", fontsize=9,
            color=COLORS["green"], fontweight="bold")

    ax.annotate("", xy=(2.85, 1.7), xytext=(2.35, 1.7),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2))
    ax.text(2.6, 2.1, f"W₂\n({h}→{d})", ha="center", fontsize=8.5, color="#555")

    fbox(3.4, "Output",  f"d = {d}",   COLORS["purple"])

    ax.text(2.2, 3.1,
            f"Params per block: 2 × {d} × {h} = {params_per:,}     |     "
            f"Total FFN ({n} blocks): {params_all:,}",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#bbb"))

    ax.set_title("FeedForward Network: 4× Expansion Creates a Rich Working Space",
                 fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────
def plot_gradient_flow(n_layers=12, dim=8, seed=42):
    """Bar charts of gradient norms with vs without residual connections."""
    torch.manual_seed(seed)

    class NoSkip(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        def forward(self, x):
            for l in self.layers: x = torch.tanh(l(x))
            return x.sum()

    class WithSkip(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        def forward(self, x):
            for l in self.layers: x = x + torch.tanh(l(x))
            return x.sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, Model, name, color in [
        (axes[0], NoSkip,   "No Skip Connections",   COLORS["red"]),
        (axes[1], WithSkip, "With Skip Connections",  COLORS["green"]),
    ]:
        m = Model(); x = torch.randn(4, dim, requires_grad=True)
        m(x).backward()
        norms = [p.grad.norm().item() for p in m.parameters() if p.grad is not None]
        ax.bar(range(len(norms)), norms, color=color, alpha=0.85, edgecolor="white")
        mn = np.mean(norms)
        ax.axhline(mn, color="black", lw=1.5, linestyle="--", label=f"mean={mn:.4f}")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Parameter group"); ax.set_ylabel("Gradient norm")
        ax.legend()

    plt.suptitle("Residual Connections: A Highway for Gradients Through Deep Networks",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────
def trace_gpt_shapes(config, B=2, T=10, print_output=True):
    """Return (and optionally print) a table of tensor shapes at every GPT stage."""
    C = config["emb_dim"]; V = config["vocab_size"]; n = config["n_layers"]
    stages = [
        ("Input token IDs",              (B, T)),
        ("After tok_emb lookup",         (B, T, C)),
        ("After pos_emb lookup",         (T, C)),
        ("Input embedding  (tok + pos)", (B, T, C)),
        ("After Dropout",                (B, T, C)),
    ]
    for i in range(n):
        stages.append((f"After TransformerBlock {i+1:2d}", (B, T, C)))
    stages += [
        ("After Final LayerNorm",        (B, T, C)),
        ("Logits (output linear)",       (B, T, V)),
    ]
    if print_output:
        print(f"{'Stage':<42}  {'Shape':>26}")
        print("─" * 72)
        for name, shape in stages:
            marker = "  ◀  OUTPUT" if "Logits" in name else ""
            print(f"{name:<42}  {str(shape):>26}{marker}")
    return stages


def plot_shape_journey(config, B=2, T=10):
    """Log-scale bar chart tracing tensor dim at every GPT stage."""
    stages = trace_gpt_shapes(config, B, T, print_output=False)
    labels = [s[0].replace("After ", "").replace("TransformerBlock", "Block") for s, _ in
              [(s, sh) for s, sh in stages]]
    dims   = [sh[-1] for _, sh in stages]

    base_c = COLORS
    colors = (
        ["#e74c3c"] +           # input ids
        [COLORS["orange"]] * 2 + # embeddings
        [COLORS["yellow"]] +     # combined
        [COLORS["gray"]] +       # dropout
        [COLORS["blue"]] * config["n_layers"] +  # blocks
        [COLORS["green"]] +      # final norm
        [COLORS["purple"]]       # logits
    )

    fig, ax = plt.subplots(figsize=(18, 5))
    x_pos = np.arange(len(labels))
    bars  = ax.bar(x_pos, dims, color=colors, edgecolor="white", lw=1.0, alpha=0.9)

    for bar, d in zip(bars, dims):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f"{d:,}", ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=55)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace("Input token IDs","Token IDs").replace("Input embedding  (tok + pos)","Input Emb")
                        for s in labels], rotation=55, ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("Size of last dimension (log scale)", fontsize=11)
    ax.set_title(f"Tensor Shape Journey Through GPT-2 Small  (B={B}, T={T})",
                 fontsize=13, fontweight="bold")

    legend_items = [
        mpatches.Patch(color=COLORS["orange"],  label="Embedding layers"),
        mpatches.Patch(color=COLORS["yellow"],  label="Input embedding (combined)"),
        mpatches.Patch(color=COLORS["blue"],    label=f"Transformer blocks ×{config['n_layers']}  (constant d={config['emb_dim']})"),
        mpatches.Patch(color=COLORS["green"],   label="Final LayerNorm"),
        mpatches.Patch(color=COLORS["purple"],  label=f"Output logits  (vocab={config['vocab_size']})"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9)
    plt.tight_layout(); plt.show()
    print(f"The embedding dimension stays CONSTANT at {config['emb_dim']} through all "
          f"{config['n_layers']} blocks. Only the final projection expands it to {config['vocab_size']}.")


# ────────────────────────────────────────────────────────────────────────────
def plot_parameter_breakdown(model):
    """Pie + bar chart of parameters per top-level component."""
    breakdown = {name: sum(p.numel() for p in mod.parameters())
                 for name, mod in model.named_children() if sum(p.numel() for p in mod.parameters()) > 0}
    label_map = {
        "tok_emb": "Token Embedding",
        "pos_emb": "Positional Embedding",
        "blocks":  "Transformer Blocks ×12",
        "norm":    "Final LayerNorm",
        "head":    "Output Linear (LM head)\n[weight-tied → 0 extra]",
    }
    total = sum(breakdown.values())

    print(f"{'Component':<32}  {'Params':>14}  {'Share':>7}")
    print("─" * 58)
    for k, v in breakdown.items():
        print(f"{label_map.get(k, k):<32}  {v:>14,}  {100*v/total:>6.2f}%")
    print("─" * 58)
    print(f"{'TOTAL':<32}  {total:>14,}  100.00%")

    items   = {label_map.get(k, k): v for k, v in breakdown.items()}
    names   = list(items.keys())
    vals    = list(items.values())
    palette = [COLORS["orange"], COLORS["yellow"], COLORS["blue"],
               COLORS["green"], COLORS["gray"]][:len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bars = axes[0].barh(names, [v/1e6 for v in vals],
                        color=palette, edgecolor="white", lw=1.5, alpha=0.9)
    axes[0].set_xlabel("Parameters (millions)"); axes[0].set_title("Params by Component", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{v/1e6:.1f}M", va="center", fontsize=10)

    axes[1].pie(vals, labels=[n.replace("\n[weight-tied → 0 extra]","") for n in names],
                colors=palette, autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", lw=2))
    axes[1].set_title("Parameter Distribution", fontsize=12, fontweight="bold")

    plt.suptitle(f"GPT-2 Small — {total/1e6:.1f}M Parameters", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()


# ────────────────────────────────────────────────────────────────────────────
def plot_gpt_variants(configs: dict, ModelClass):
    """Bar charts comparing the GPT-2 family."""
    param_counts = {}
    for name, cfg in configs.items():
        m = ModelClass(cfg)
        param_counts[name] = sum(p.numel() for p in m.parameters()) / 1e6
        del m

    names  = list(configs.keys())
    layers = [c["n_layers"]  for c in configs.values()]
    dims   = [c["emb_dim"]   for c in configs.values()]
    params = list(param_counts.values())
    palette = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (vals, title, unit) in zip(axes, [
        (params, "Total Parameters",     "M params"),
        (layers, "Number of Layers",     "layers"),
        (dims,   "Embedding Dimension",  "d_model"),
    ]):
        bars = ax.bar(range(4), vals, color=palette, edgecolor="white", lw=1.5, alpha=0.9)
        ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold"); ax.set_ylabel(unit)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("The GPT-2 Family — Same Architecture, Different Scale",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()
