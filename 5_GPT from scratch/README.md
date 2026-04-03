# Module 5: GPT from Scratch 🧠

**Building a Large Language Model — No Shortcuts, No Black Boxes**

**📍 Location:** `5_GPT from scratch/`  
**🎯 Prerequisite:** [Module 4: Applied Deep Learning with PyTorch](../4_Applied%20Deep%20Learning%20with%20PyTorch/README.md)  
**➡️ Next Step:** Capstone / final integration project

Welcome to **Module 5** of **SAIR** — the deepest technical challenge of the entire track. You will build a GPT-style language model from absolute scratch: every tokenizer, every attention head, every training loop, and every fine-tuning step. By the end you will have a working LLM trained on real text and fine-tuned for two downstream tasks.

No HuggingFace `from_pretrained`. No shortcuts. Just PyTorch and first principles.

---

## 🎯 Module Overview

This module is structured in three parts:

1. **4 core build notebooks** — tokenizer, attention, GPT architecture, and training
2. **1 advanced training extension** — deeper training experiments and refinements
3. **2 SFT notebooks** — classification and instruction-following fine-tuning

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
    ▼
[ Notebook 5 ] Advanced Training Extension
    │
    ├──▶ [ Notebook 6 ] SFT — Text Classification
    │
    └──▶ [ Notebook 7 ] SFT — Instruction Following
```

---

## 📚 Notebook Breakdown

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

### Notebook 5 — Advanced Training Extension
📓 `5.TRAIN_Pro.ipynb`

An additional training notebook that extends the core GPT training workflow with more advanced experimentation before fine-tuning.

| Topic | What You Build |
|-------|---------------|
| Training refinements | Extend the base training workflow beyond the first end-to-end run |
| Deeper experimentation | Inspect more training behaviour and practical tradeoffs |
| Bridge to SFT | Prepare the model and workflow for downstream adaptation tasks |

---

### Notebook 6 — SFT: Text Classification
📓 `6.SFT_Text_Classification.ipynb`

Fine-tune the pretrained GPT on a labelled classification dataset.

| Topic | Details |
|-------|---------|
| Task | Sequence-level text classification |
| Approach | Replace the LM head with a classification head |
| Training | Supervised fine-tuning on labelled examples |
| Evaluation | Accuracy, F1, confusion matrix |

---

### Notebook 7 — SFT: Instruction Following
📓 `7.SFT_Instruction_Following.ipynb`

Fine-tune the pretrained GPT to follow instructions in a prompt-response format.

| Topic | Details |
|-------|---------|
| Task | Instruction following / chat-style generation |
| Format | `[INSTRUCTION] ... [RESPONSE] ...` prompt template |
| Approach | Supervised fine-tuning on instruction-response pairs |
| Evaluation | Qualitative generation + loss on response tokens only |

---

## 🏗️ What Is In The Repo Right Now

This module currently ships as a notebook-first learning path plus prepared data files and reading resources.

There is **not yet** a `pipeline/` package in this folder. When that production reimplementation is added later, it should be documented as a separate deliverable rather than mixed into the current runnable instructions.

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

# Continue through the remaining notebooks in order
uv run jupyter notebook 2.ATTENTION.ipynb
uv run jupyter notebook 3.GPT.ipynb
uv run jupyter notebook 4.TRAIN.ipynb
uv run jupyter notebook 5.TRAIN_Pro.ipynb
uv run jupyter notebook 6.SFT_Text_Classification.ipynb
uv run jupyter notebook 7.SFT_Instruction_Following.ipynb
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
├── 5.TRAIN_Pro.ipynb               # Advanced training extension
├── 6.SFT_Text_Classification.ipynb # Notebook 6 — classification fine-tuning
├── 7.SFT_Instruction_Following.ipynb # Notebook 7 — instruction fine-tuning
├── data/
│   ├── train_ids.bin               # ~1.9M tokens (90% of corpus)
│   ├── val_ids.bin                 # ~148K tokens (7%)
│   └── test_ids.bin                # ~61K tokens (3%)
├── Resources/
│   ├── raschka_llm_from_scratch.pdf          # Primary textbook
│   ├── raschka_llm_from_scratch_cover.png    # Book cover
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
<img src="Resources/raschka_llm_from_scratch_cover.jpg" alt="Build a Large Language Model From Scratch — Sebastian Raschka" width="220"/>
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
