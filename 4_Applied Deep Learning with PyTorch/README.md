# Module 4: Applied Deep Learning with PyTorch ⚡

**From PyTorch Fundamentals to Modern Gen AI**

**📍 Location:** `4_Applied Deep Learning with PyTorch/`
**🎯 Prerequisite:** [Module 3: Neural Networks from Scratch](../3_Neural%20Network%20from%20scratch/README.md)
**➡️ Next Module:** [Module 5: GPT from Scratch](../5_GPT%20from%20scratch/README.md)

Welcome to **Module 4** of **SAIR** – your comprehensive journey into applied deep learning with PyTorch. This module bridges theory and practice, taking you from tensor operations all the way to modern architectures, with stops along the way for CNNs, YOLOv8, RNNs, LSTMs, and HuggingFace transformers.

---

## 🎯 Module Overview

This module is structured in four progressive sections:

1. **PyTorch Fundamentals** – Tensors, autograd, training loops
2. **Computer Vision with CNNs** – From scratch to YOLOv8 and ViTs
3. **Sequence Modeling & NLP** – RNNs, LSTMs, HuggingFace, fine-tuning
4. **Classification Hub** – Five open-ended projects across all modalities

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![UV](https://img.shields.io/badge/UV-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

</div>

---

## 📚 Module Contents

### **1 — PyTorch Fundamentals**
📁 `1_PyTorch Fundemntals/`

| Notebook | Focus |
|----------|-------|
| `1_Intro.ipynb` | Tensors, autograd, model building, training loops |
| `2_DataLoader.ipynb` | Dataset classes, DataLoader optimization, performance |

Labs: `labs/lab_1.ipynb`, `labs/lab_2.ipynb`

---

### **2 — Computer Vision & CNNs**
📁 `2_Vision and CNN/`

| Notebook | Focus |
|----------|-------|
| `3_CNN.ipynb` | Convolutional Neural Networks from scratch |
| `4_Transfer_and_ResNet.ipynb` | Transfer learning, ResNet, pretrained models |
| `5A_YOLO.ipynb` | YOLOv8 object detection |
| `5B_Segment_Pose.ipynb` | Instance segmentation and pose estimation |
| `5C_ViTs_and_Deploy.ipynb` | Vision Transformers and model deployment |

Labs: `labs/lab_3.ipynb`, `labs/lab_4.ipynb`

**Production Demos** — `2_Vision and CNN/Demos/`

| Demo | Description |
|------|-------------|
| `demo_01_live_detection.py` | Real-time object detection with webcam |
| `demo_02_background_removal.py` | Background removal via segmentation |
| `demo_03_pose_estimation.py` | Human pose estimation in real-time |
| `demo_04_gesture_control.py` | Control applications with hand gestures |
| `demo_05_model_comparison.py` | Compare YOLOv8n/s/m performance |
| `demo_06_batch_processing.py` | Process multiple images efficiently |
| `demo_07_video_processing.py` | Object detection on video files |

```bash
cd '2_Vision and CNN/Demos'
uv pip install -r requirements.txt
python run_demos.py
```

Pre-trained models in `2_Vision and CNN/`:
`yolov8n.pt` · `yolov8n-seg.pt` · `yolov8n-pose.pt` · `yolov8n.onnx`
`best_x.pt` · `best_yolo26n_100.pt` · `best_yolo26m_100.pt` · `yassir_best.pt`

---

### **3 — Sequence Modeling & NLP**
📁 `3_Sequence and NLP/`

| Notebook | Focus |
|----------|-------|
| `6_Intro_to_Seq.ipynb` | RNNs, LSTMs, sequence modeling fundamentals |
| `7_Seq_to_Seq.ipynb` | Sequence-to-sequence architectures |
| `8A_HuggingFace_Ecosystem.ipynb` | Pipelines, tokenizers, models, datasets |
| `8B_Hugging_Face_Finetuning.ipynb` | Fine-tuning pretrained transformers end to end |

Saved models: `best_rnnclassifier.pt`, `best_lstmclassifier.pt`

Training data: `harry_potter_txt/` — all 7 books used for sequence modeling experiments

**Text Classification Pipeline** — `3_Sequence and NLP/Text Classification/`

A production-style NLP project demonstrating five approaches to text classification on the same dataset:

| Notebook | Approach |
|----------|----------|
| `notebooks/01_eda.ipynb` | Exploratory data analysis |
| `notebooks/02_feature_extraction.ipynb` | TF-IDF and classical ML features |
| `notebooks/03_embedding.ipynb` | Sentence embeddings pipeline |
| `notebooks/04_finetune.ipynb` | Full transformer fine-tuning |
| `notebooks/05_prompt.ipynb` | Zero-shot and prompt-based classification |

Fully modular — `src/` contains separate modules for data, training, evaluation, and inference.
Orchestrated by `run_pipeline.py`. Deployable app at `app/app.py`.

```bash
cd '3_Sequence and NLP/Text Classification'
uv pip install -r requirements.txt
python run_pipeline.py
```

---

### **4 — Classification Hub**
📁 `Classification Hub/`

Five open-ended project notebooks — one per data modality.
No steps. No guided cells. A problem, a dataset, and a blank notebook.

| Project | Modality | Task |
|---------|----------|------|
| `Ex_1_Tabular_Classification.ipynb` | 📊 Tabular | Rice type classifier from grain measurements |
| `Ex_2_Image_Classification.ipynb` | 🖼️ Image (scratch) | Animal face classifier with a custom CNN |
| `Ex_3_Image_Classification_Pretrained.ipynb` | 🌿 Image (pretrained) | Bean leaf disease detector via transfer learning |
| `Ex_4_Audio_Classification.ipynb` | 🎵 Audio | Quran reciter identifier |
| `Ex_5_Text_Classification_Transformers.ipynb` | 📝 Text | Sarcasm detector with a fine-tuned transformer |

See `Classification Hub/README.md` for what each submission must include.

---

## 📂 Directory Structure

```
4_Applied Deep Learning with PyTorch/
│
├── 1_PyTorch Fundemntals/
│   ├── 1_Intro.ipynb
│   ├── 2_DataLoader.ipynb
│   └── labs/
│       ├── lab_1.ipynb
│       └── lab_2.ipynb
│
├── 2_Vision and CNN/
│   ├── 3_CNN.ipynb
│   ├── 4_Transfer_and_ResNet.ipynb
│   ├── 5A_YOLO.ipynb
│   ├── 5B_Segment_Pose.ipynb
│   ├── 5C_ViTs_and_Deploy.ipynb
│   ├── assets/
│   ├── data/
│   ├── datasets/
│   ├── generated/
│   ├── models/
│   ├── labs/
│   │   ├── lab_3.ipynb
│   │   └── lab_4.ipynb
│   ├── Demos/
│   │   ├── demo_01_live_detection.py
│   │   ├── demo_02_background_removal.py
│   │   ├── demo_03_pose_estimation.py
│   │   ├── demo_04_gesture_control.py
│   │   ├── demo_05_model_comparison.py
│   │   ├── demo_06_batch_processing.py
│   │   ├── demo_07_video_processing.py
│   │   ├── run_demos.py
│   │   ├── requirements.txt
│   │   └── README_DEMOS.md
│   ├── street.jpg
│   ├── coco128.yaml
│   ├── yolov8n.pt
│   ├── yolov8n-seg.pt
│   ├── yolov8n-pose.pt
│   ├── yolov8n.onnx
│   ├── best_x.pt
│   ├── best_yolo26n_100.pt
│   ├── best_yolo26m_100.pt
│   ├── best_yolo26n_50.pt
│   ├── best_yolov8n_100.pt
│   └── yassir_best.pt
│
├── 3_Sequence and NLP/
│   ├── 6_Intro_to_Seq.ipynb
│   ├── 7_Seq_to_Seq.ipynb
│   ├── 8A_HuggingFace_Ecosystem.ipynb
│   ├── 8B_Hugging_Face_Finetuning.ipynb
│   ├── assets/
│   │   ├── rnn.png
│   │   ├── rnns.png
│   │   └── lstm.png
│   ├── best_rnnclassifier.pt
│   ├── best_lstmclassifier.pt
│   ├── harry_potter_txt/
│   │   ├── Book 1 - The Philosopher's Stone.txt
│   │   ├── Book 2 - The Chamber of Secrets.txt
│   │   ├── Book 3 - The Prisoner of Azkaban.txt
│   │   ├── Book 4 - The Goblet of Fire.txt
│   │   ├── Book 5 - The Order of the Phoenix.txt
│   │   ├── Book 6 - The Half Blood Prince.txt
│   │   └── Book 7 - The Deathly Hallows.txt
│   └── Text Classification/
│       ├── app/
│       │   └── app.py
│       ├── config.py
│       ├── models/
│       │   └── finetuned/
│       ├── notebooks/
│       │   ├── 01_eda.ipynb
│       │   ├── 02_feature_extraction.ipynb
│       │   ├── 03_embedding.ipynb
│       │   ├── 04_finetune.ipynb
│       │   └── 05_prompt.ipynb
│       ├── src/
│       │   ├── data.py
│       │   ├── embedding_pipeline.py
│       │   ├── eval.py
│       │   ├── features.py
│       │   ├── train_features.py
│       │   ├── train_finetune.py
│       │   └── zero_shot.py
│       ├── run_pipeline.py
│       └── requirements.txt
│
├── Classification Hub/
│   ├── Ex_1_Tabular_Classification.ipynb
│   ├── Ex_2_Image_Classification.ipynb
│   ├── Ex_3_Image_Classification_Pretrained.ipynb
│   ├── Ex_4_Audio_Classification.ipynb
│   ├── Ex_5_Text_Classification_Transformers.ipynb
│   └── README.md
│
├── data/
├── datasets/
├── detection_output/
├── lab_assignments/
│   └── [student_name]/
├── papers/
│   ├── AlexNet_paper.pdf
│   └── ResNet_paper.pdf
├── coco128.yaml
├── coco128_dataset.yaml
├── yolov8n.pt
├── yolov8n-seg.pt
├── yolov8n-pose.pt
├── yolov8n.onnx
└── README.md
```

---

## 🚀 Learning Pathway

### **Phase 1: Foundations** (Week 1–2)
1. `1_PyTorch Fundemntals/1_Intro.ipynb` – Tensors and autograd
2. Complete `labs/lab_1.ipynb`
3. `1_PyTorch Fundemntals/2_DataLoader.ipynb` – Data pipelines
4. Complete `labs/lab_2.ipynb`

### **Phase 2: Computer Vision** (Week 3–4)
1. `2_Vision and CNN/3_CNN.ipynb` – Build CNNs from scratch
2. Complete `labs/lab_3.ipynb`
3. `2_Vision and CNN/4_Transfer_and_ResNet.ipynb` – Transfer learning
4. Complete `labs/lab_4.ipynb`
5. `5A_YOLO.ipynb` → `5B_Segment_Pose.ipynb` → `5C_ViTs_and_Deploy.ipynb`
6. Run the Demos in `2_Vision and CNN/Demos/`

### **Phase 3: Sequence Modeling & NLP** (Week 5–6)
1. `3_Sequence and NLP/6_Intro_to_Seq.ipynb` – RNNs and LSTMs
2. `3_Sequence and NLP/7_Seq_to_Seq.ipynb` – Sequence-to-sequence
3. `3_Sequence and NLP/8A_HuggingFace_Ecosystem.ipynb` – The HuggingFace stack
4. `3_Sequence and NLP/8B_Hugging_Face_Finetuning.ipynb` – Fine-tuning
5. Explore the Text Classification production pipeline

### **Phase 4: Classification Hub** (Ongoing)
Work through all five projects independently. No guidance — just the problem brief and the dataset.

### **Phase 5: GPT from Scratch** → Module 5
Continue to `5_GPT from scratch/` — a standalone module dedicated to building a GPT-style language model end to end.

---

## 🎯 Learning Outcomes

After completing this module, you will be able to:

- **Build** neural networks from scratch using PyTorch
- **Design** efficient data pipelines with custom Datasets and DataLoaders
- **Train** CNNs for image classification
- **Deploy** YOLOv8 for detection, segmentation, and pose estimation
- **Build** sequence models with RNNs and LSTMs
- **Use** the HuggingFace ecosystem end to end
- **Fine-tune** pretrained transformers for downstream tasks
- **Apply** your skills independently across all five major data modalities

---

## 🔧 Installation & Setup with UV

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to module
cd 'SAIR/4_Applied Deep Learning with PyTorch'

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Core dependencies
uv pip install torch torchvision torchaudio
uv pip install jupyter matplotlib numpy pandas tqdm

# For NLP and HuggingFace
uv pip install transformers datasets accelerate

# For YOLOv8 demos
cd '2_Vision and CNN/Demos'
uv pip install -r requirements.txt

# For the Text Classification pipeline
cd '../../3_Sequence and NLP/Text Classification'
uv pip install -r requirements.txt

# Launch Jupyter
cd ../..
jupyter notebook
```

### **UV Commands Cheat Sheet**

| Command | Purpose |
|---------|---------|
| `uv venv` | Create virtual environment |
| `uv pip install <package>` | Install a package |
| `uv pip install -r requirements.txt` | Install from requirements file |
| `uv pip list` | List installed packages |
| `uv pip freeze > requirements.txt` | Generate requirements file |
| `uv pip uninstall <package>` | Remove a package |
| `uv cache clean` | Clean uv cache |

---

## 📝 Notes

- **GPU**: All notebooks detect CUDA automatically. Check with `torch.cuda.is_available()`.
- **Lab Submissions**: Place completed labs in `lab_assignments/[your_name]/`
- **Classification Hub**: Open-ended projects. Read the brief, build the solution.
- **GPT Deep Dive**: Covered in Module 5 (`5_GPT from scratch/`).
- **Model Files**: Saved YOLO and RNN/LSTM models are included and ready to use.
- **UV Speed**: UV is significantly faster than pip. ⚡

---

## 📚 Additional Resources

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [HuggingFace Documentation](https://huggingface.co/docs)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [Papers with Code](https://paperswithcode.com/)

> *"From tensors to production – understanding every layer of the stack."*

**Happy Learning with UV! 🚀⚡**
