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
📓 `1.DATA.ipynb` · **⏱ 4–5 hours**

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
📓 `2.ATTENTION.ipynb` · **⏱ 6–7 hours**

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
📓 `3.GPT.ipynb` · **⏱ 6–8 hours**

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
📓 `4.TRAIN.ipynb` · **⏱ 6–8 hours** (+ training time)

| Topic | What You Build |
|-------|---------------|
| Cross-entropy loss | Next-token prediction objective |
| AdamW optimiser | With weight decay and gradient clipping |
| Training loop | Epoch loop, loss tracking, val loss monitoring |
| Text generation | Greedy decoding and temperature sampling |
| Checkpointing | Save and reload model weights |

---

### Lecture 5 — SFT: Text Classification
📓 `5.SFT_Text_Classification.ipynb` · **⏱ 4–5 hours**

Fine-tune the pretrained GPT on a labelled classification dataset.

| Topic | Details |
|-------|---------|
| Task | Sequence-level text classification |
| Approach | Replace the LM head with a classification head |
| Training | Supervised fine-tuning on labelled examples |
| Evaluation | Accuracy, F1, confusion matrix |

---

### Lecture 6 — SFT: Instruction Following
📓 `6.SFT_Instruction_Following.ipynb` · **⏱ 4–5 hours**

Fine-tune the pretrained GPT to follow instructions in a prompt-response format.

| Topic | Details |
|-------|---------|
| Task | Instruction following / chat-style generation |
| Format | `[INSTRUCTION] ... [RESPONSE] ...` prompt template |
| Approach | Supervised fine-tuning on instruction-response pairs |
| Evaluation | Qualitative generation + loss on response tokens only |

---

## 💻 Compute Requirements

| Task | CPU Only | T4 GPU (Colab) | GPU 8 GB+ |
|------|----------|----------------|-----------|
| Lectures 1–3 (no training) | ✅ Fine | ✅ Fast | ✅ Fast |
| Lecture 4 — full training run | ⚠️ 12–24 hrs | ~2–3 hrs/epoch | ~30–60 min/epoch |
| Lecture 5 — SFT classification | ⚠️ Slow | ~30–60 min | ~10–15 min |
| Lecture 6 — SFT instruction | ⚠️ Slow | ~45–90 min | ~15–30 min |

> **Recommended:** Google Colab T4 GPU (free tier) is sufficient for all lectures.  
> For Lecture 4 on CPU: reduce `GPT_CONFIG` to a smaller model (e.g., 4 layers, 256 dim) to test your loop first.

---

## 🔧 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` | Batch size too large | Reduce `batch_size` in config; try 4 or 8 |
| Generation produces only repetitions | Untrained weights or greedy decoding | Use temperature sampling: `temperature=1.0, top_k=50` |
| Loss not decreasing after epoch 1 | Learning rate too high | Try `lr=3e-4` with `AdamW` and warmup |
| `AssertionError: shape mismatch` in weight loading | Wrong `qkv_bias` setting | Set `qkv_bias=True` when loading pretrained GPT-2 weights |
| `tiktoken` encoding error | Non-UTF-8 characters in corpus | Strip with `text.encode('utf-8', errors='ignore').decode()` |
| Checkpoint not found | Wrong path or never saved | Check `config.py` for `checkpoint_dir`; run training first |

---

## 🏗️ Modular Pipeline
📁 `pipeline/`

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
├── Resources/
│   ├── raschka_llm_from_scratch.pdf          # Primary textbook
│   ├── raschka_llm_from_scratch_cover.jpg    # Book cover
│   ├── raschka_llm_exercises.pdf             # Exercise companion
│   └── attention_is_all_you_need.pdf         # Vaswani et al. 2017
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

## 📚 Resources

All materials are in `Resources/` — read in parallel with the lectures, not after.

---

### 📖 Primary Textbook — Read in Parallel

<div align="center">

<a href="Resources/raschka_llm_from_scratch.pdf">
<img src="Resources/raschka_llm_from_scratch_cover.png" alt="Build a Large Language Model From Scratch — Sebastian Raschka" width="220"/>
</a>

**Build a Large Language Model (From Scratch)**  
*Sebastian Raschka — Manning Publications, 2024*

</div>

This module is built around this book. Every lecture maps directly to a chapter:

| Lecture | Book Chapter |
|---------|-------------|
| Lecture 1 — Data & Tokenization | Ch. 2: Working with text data |
| Lecture 2 — Attention | Ch. 3: Coding attention mechanisms |
| Lecture 3 — GPT Architecture | Ch. 4: Implementing a GPT model from scratch |
| Lecture 4 — Training | Ch. 5: Pretraining on unlabeled data |
| Lecture 5 — SFT: Classification | Ch. 6: Fine-tuning for classification |
| Lecture 6 — SFT: Instruction Following | Ch. 7: Fine-tuning to follow instructions |

Read each chapter **before** its lecture. The notebook is your hands-on implementation of what the book explains.

---

### 🧪 Exercise Book — Test Your Understanding

<a href="Resources/raschka_llm_exercises.pdf">**LLM From Scratch — Exercise Companion**</a> *(same author)*

After each lecture, work through the corresponding exercises. If you can answer them without looking at your notes, you own the material.

---

### 📄 Core Paper

<a href="Resources/attention_is_all_you_need.pdf">**Attention Is All You Need**</a> — Vaswani et al., 2017

The paper that started everything. Read it alongside Lecture 2 (Attention). The notation in your notebooks maps directly to the equations in this paper.

---

### 🔗 Additional References

| Resource | When to Use |
|----------|------------|
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | After Lecture 4 — understand why scale matters |
| [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT) | After finishing all lectures — compare implementations |

---

> *"You haven't truly understood a model until you've trained one yourself."*

**Module 5 of the SAIR Jr. Certification Track 🇸🇩**
