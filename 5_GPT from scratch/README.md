# Module 5: GPT from Scratch 🧠

**Building a Large Language Model — No Shortcuts, No Black Boxes**

**📍 Location:** `5_GPT from scratch/`  
**🎯 Prerequisite:** [Module 4: Applied Deep Learning with PyTorch](../4_Applied%20Deep%20Learning%20with%20PyTorch/README.md)  
**➡️ Next Module:** Capstone (coming soon)

Welcome to **Module 5** of **SAIR** — the deepest technical challenge of the entire track. You will build a GPT-style language model from absolute scratch: every tokenizer, every attention head, every training loop, and every fine-tuning step. By the end you will have a working LLM trained on real text and fine-tuned for two downstream tasks.

No HuggingFace `from_pretrained`. No shortcuts. Just PyTorch and first principles.

---

## 🎯 Module Overview

This module is structured in two parts:

1. **6 Lectures** — progressive notebooks building every component from the ground up
2. **Modular Pipeline** — a production-style reimplementation of the full stack as clean, importable Python modules

### What You Will Build

```
Raw Text (Harry Potter corpus)
    │
    ▼
[ Lecture 1 ] Tokenizer + DataLoader
    │
    ▼
[ Lecture 2 ] Attention Mechanism (dot-product → causal → multi-head)
    │
    ▼
[ Lecture 3 ] Full GPT Architecture (LayerNorm, GELU, FFN, TransformerBlock)
    │
    ▼
[ Lecture 4 ] Training Loop (loss, optimiser, checkpointing, generation)
    │
    ├──▶ [ Lecture 5 ] SFT — Text Classification
    │
    └──▶ [ Lecture 6 ] SFT — Instruction Following
```

---

## 📚 Lecture Breakdown

### Lecture 1 — Data Preparation & Tokenization
📓 `1.DATA.ipynb`

| Topic | What You Build |
|-------|---------------|
| Corpus loading | Read all 7 Harry Potter books from local `.txt` files |
| Custom tokenizer | `TokenizerV1` and `TokenizerV2` from scratch (regex-based) |
| Production tokenizer | `tiktoken` BPE — same tokenizer used in GPT-2 and GPT-4 |
| Sliding window | Input/target pair generation for next-token prediction |
| Binary storage | Train / val / test splits saved as `.bin` files |
| DataLoader | `GPT2Dataset` + PyTorch `DataLoader` ready for training |

**Corpus:** ~1.9M training tokens, 148K validation tokens

---

### Lecture 2 — Attention Mechanisms
📓 `2.ATTENTION.ipynb`

| Topic | What You Build |
|-------|---------------|
| Dot-product attention | From raw scores to context vector, step by step |
| Softmax & scaling | Why we divide by √d_k and what goes wrong without it |
| Learnable Q, K, V | `AttentionV1` with `nn.Parameter`, `AttentionV2` with `nn.Linear` |
| Causal masking | Upper-triangular mask preventing information leakage |
| Dropout | Regularisation inside the attention layer |
| Multi-head attention | Full `MultiHeadAttention` module — the exact class used in GPT |

**Key insight:** every plot and diagram uses the same `["Your", "journey", "starts", "with", "one", "step"]` token sequence so you can trace every operation back to concrete numbers.

---

### Lecture 3 — GPT Architecture
📓 `3.GPT.ipynb`

| Component | Details |
|-----------|---------|
| `LayerNorm` | From scratch — learnable scale & shift, Pre-LN design |
| `GELU` | Exact tanh approximation used in GPT-2 |
| `FeedForward` | 2-layer MLP with 4× hidden dimension expansion |
| `TransformerBlock` | Attention + FFN + residual connections + Pre-LN |
| `GPTModel` | Token embeddings + positional embeddings + 12 stacked blocks |
| Configuration | `GPT_CONFIG_124M` matching GPT-2 Small exactly |

**Result:** a fully functional 124M-parameter GPT-2 Small you can run inference on.

---

### Lecture 4 — Training Loop
📓 `4.TRAIN.ipynb`

| Topic | What You Build |
|-------|---------------|
| Cross-entropy loss | Next-token prediction objective |
| AdamW optimiser | With weight decay and gradient clipping |
| Training loop | Epoch loop, loss tracking, val loss monitoring |
| Text generation | Greedy decoding and temperature sampling |
| Checkpointing | Save and reload model weights |

---

### Lecture 5 — SFT: Text Classification
📓 `5.SFT_Text_Classification.ipynb`

Fine-tune the pretrained GPT on a labelled classification dataset.

| Topic | Details |
|-------|---------|
| Task | Sequence-level text classification |
| Approach | Replace the LM head with a classification head |
| Training | Supervised fine-tuning on labelled examples |
| Evaluation | Accuracy, F1, confusion matrix |

---

### Lecture 6 — SFT: Instruction Following
📓 `6.SFT_Instruction_Following.ipynb`

Fine-tune the pretrained GPT to follow instructions in a prompt-response format.

| Topic | Details |
|-------|---------|
| Task | Instruction following / chat-style generation |
| Format | `[INSTRUCTION] ... [RESPONSE] ...` prompt template |
| Approach | Supervised fine-tuning on instruction-response pairs |
| Evaluation | Qualitative generation + loss on response tokens only |

---

## 🏗️ Modular Pipeline
📁 `pipeline/` *(coming soon)*

The same implementation from the lectures, restructured as a production-ready Python package:

```
pipeline/
├── data/
│   ├── loader.py          # book loading + train/val/test split
│   ├── tokenizer.py       # tiktoken wrapper
│   └── dataset.py         # GPT2Dataset + DataLoader factory
├── model/
│   ├── attention.py       # MultiHeadAttention
│   ├── blocks.py          # TransformerBlock, LayerNorm, GELU, FeedForward
│   └── gpt.py             # GPTModel + config
├── training/
│   ├── trainer.py         # training loop, checkpointing
│   └── generate.py        # text generation utilities
├── sft/
│   ├── classifier.py      # classification fine-tuning
│   └── instruct.py        # instruction fine-tuning
├── config.py              # GPT_CONFIG_124M and variants
└── run.py                 # single entry point: train / generate / fine-tune
```

The pipeline is the difference between *understanding* the code and *using* it in production. Once you have completed all 6 lectures, you will reimplement everything modularly — no notebooks, no scaffolding.

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| `torch` | Everything — model, training, tensors |
| `tiktoken` | Production BPE tokenizer |
| `numpy` | Binary data storage and preprocessing |
| `matplotlib` | All visualisations in the lectures |
| Standard library only | Data loading — no HuggingFace datasets |

---

## 🚀 Getting Started

```bash
# From the SAIR root
cd "5_GPT from scratch"

# Run a lecture notebook
uv run jupyter notebook 1.DATA.ipynb

# Or run the full pipeline (after completing lectures)
uv run python pipeline/run.py --mode train
uv run python pipeline/run.py --mode generate --prompt "Harry looked at"
```

**Data:** the Harry Potter books are already in the repo at  
`4_Applied Deep Learning with PyTorch/3_Sequence and NLP/harry_potter_txt/`

No downloads required.

---

## 📂 Directory Structure

```
5_GPT from scratch/
├── 1.DATA.ipynb                    # Lecture 1 — tokenization & data pipeline
├── 2.ATTENTION.ipynb               # Lecture 2 — attention mechanisms
├── 3.GPT.ipynb                     # Lecture 3 — full GPT architecture
├── 4.TRAIN.ipynb                   # Lecture 4 — training loop
├── 5.SFT_Text_Classification.ipynb # Lecture 5 — classification fine-tuning
├── 6.SFT_Instruction_Following.ipynb # Lecture 6 — instruction fine-tuning
├── data/
│   ├── train_ids.bin               # ~1.9M tokens (90% of corpus)
│   ├── val_ids.bin                 # ~148K tokens (7%)
│   └── test_ids.bin                # ~61K tokens (3%)
├── pipeline/                       # Modular production implementation
└── README.md
```

---

## 🎯 Learning Outcomes

After completing this module you will be able to:

- **Explain** how every component of a transformer works — not just use it
- **Implement** multi-head causal attention from a blank file
- **Build** the full GPT-2 architecture matching published parameter counts
- **Train** a language model on a real corpus from scratch
- **Fine-tune** a pretrained LLM for classification and instruction following
- **Architect** a modular ML pipeline separating data, model, training, and serving

---

## 📚 Recommended Reading

| Resource | Why |
|----------|-----|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The original transformer paper — read alongside Lecture 2 |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | GPT-3 paper — context for why this architecture matters |
| [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT) | Reference implementation — compare after you finish the lectures |

---

> *"You haven't truly understood a model until you've trained one yourself."*

**Module 5 of the SAIR Jr. Certification Track 🇸🇩**
