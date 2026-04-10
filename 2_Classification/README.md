# Module 2: Classification & Production Pipelines 🎯

**From Notebooks to Professional ML Systems**

**📍 Location:** `2_Classification/`  
**🎯 Prerequisite:** [Module 1: Regression Mastery](../1_Regression/README.md)  
**➡️ Next Module:** [Module 3: Neural Networks from Scratch](../3_Neural%20Network%20from%20scratch/README.md)

Welcome to the **Classification Module** of **SAIR** – where you transition from experimental notebooks to **production-ready ML systems** with professional pipelines and deployment architecture.

---

## 🎯 Is This Module For You?

### ✅ **Complete this module if:**
- You've mastered regression and want to tackle classification problems
- You're ready to build professional ML pipelines
- You want to learn industry best practices for ML systems
- You're preparing for ML engineering roles

### 🚀 **Review and continue if you're experienced:**
- You've built classification models but want production experience
- You're familiar with sklearn but want pipeline architecture skills
- You want to add MLflow and modular design to your toolkit

---

## 🛠️ Tools You'll Master

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

These **production tools** transform your ML code from experiments to enterprise-ready systems.

---

## 📚 What You'll Learn

| Lecture | Focus | Time Estimate | Mastery Level |
|---------|-------|---------------|---------------|
| **`Lecture_4.ipynb`** | Classification from Scratch | 4-5 hours | **Essential** |
| **`Lecture_5.ipynb`** | Production Pipeline System | 5-6 hours | **Professional** |
| **`Pipeline/` System** | Modular Architecture | 6-8 hours | **Industry Ready** |

## 🗺️ Your Learning Journey

### **Phase 1: Algorithm Fundamentals** 🎯
**Start with:** `Lecture_4.ipynb`
- Implement logistic regression from first principles
- Understand classification metrics and evaluation
- Build intuition for decision boundaries and probability

### **Phase 2: Pipeline Development** 🚀
**Continue with:** `Lecture_5.ipynb`
- Transform notebooks into modular code
- Learn configuration management
- Set up experiment tracking and hyperparameter tuning

### **Phase 3: Production Architecture** 📚
**Master with:** `Pipeline/` system
- Build end-to-end ML pipeline
- Implement professional project structure
- Deploy with Streamlit applications

---

## 🎯 Learning Outcomes

After completing this module, you will be able to:

| Skill | Where You Build It |
|-------|--------------------|
| Implement logistic regression from scratch | `Lecture_4.ipynb` |
| Interpret classification metrics: precision, recall, F1, ROC-AUC | `Lecture_4.ipynb` |
| Transform a notebook into a modular ML codebase | `Lecture_5.ipynb` |
| Design a configuration-driven pipeline | `Pipeline/` |
| Run grid search and hyperparameter tuning at scale | `Pipeline/models/` |
| Compare experiments in MLflow and select the best model | `Pipeline/` |
| Deploy a classifier as a Streamlit web app | `Pipeline/streamlit_app/` |

---

## 💡 Our Learning Philosophy

> **"From experimental code to production systems."**

At SAIR, we believe **modular, maintainable code separates hobby projects from professional systems**. This module teaches you to architect ML solutions that scale and can be maintained by teams.

**This is where you become an ML engineer, not just a model builder.**

---

## 🚀 Quick Start Guide

### **For Sequential Learners:**
```bash
# 1. Start with classification fundamentals
jupyter notebook Lecture_4.ipynb

# 2. Learn pipeline transformation
jupyter notebook Lecture_5.ipynb

# 3. Explore the production pipeline
cd Pipeline
python run_pipeline.py
```

### **For Pipeline-Focused Learners:**
```bash
# Dive directly into professional architecture
cd Pipeline
python run_pipeline.py

# Run the Streamlit app
uv run streamlit run streamlit_app/app.py
```

### **Run the Complete Example:**
```bash
# Test the breast cancer pipeline
python breast_cancer_pipline.py
```

---

## 🏗️ Professional Pipeline Architecture

### **🚀 Spaceship Titanic ML Pipeline Example**

The `Pipeline/` directory contains a **complete, production-ready ML system** that transforms Lecture 5 concepts into a professional codebase.

#### **Key Features:**
- ✅ **Modular Architecture** - Separate data, models, config, utils
- ✅ **Advanced Feature Engineering** - Custom transformers for domain-specific features
- ✅ **Multi-Model Training** - 7+ algorithms with systematic comparison
- ✅ **Hyperparameter Tuning** - Cross-validation and optimization
- ✅ **MLflow Experiment Tracking** - Reproducible experiments
- ✅ **Streamlit Deployment** - Interactive web application

#### **Pipeline Structure:**
```
Pipeline/
├── config/              # Configuration Management
│   ├── config.py        # Centralized settings and paths
│   └── __init__.py
├── data/                # Data Processing
│   ├── load_data.py     # Data ingestion and splitting
│   ├── preprocessing.py # Cleaning & preparation pipelines
│   ├── feature_engineering.py # Custom feature creation
│   └── raw/            # Source datasets
├── models/              # ML Modeling
│   ├── base_model.py    # Abstract base classes
│   ├── train_model.py   # Training orchestration
│   ├── evaluate_model.py # Comprehensive evaluation
│   └── hyperparameter_tuning.py # Systematic optimization
├── utils/               # Shared Utilities
│   └── mlflow_utils.py  # Experiment tracking helpers
├── streamlit_app/       # Deployment
│   └── app.py          # Web interface for predictions
└── run_pipeline.py      # Main execution script
```

### **Run the Complete Pipeline:**
```bash
cd Pipeline

# Execute full pipeline
python run_pipeline.py --mode full

# Or run specific steps
python run_pipeline.py --mode preprocessing    # Data only
python run_pipeline.py --mode training        # Models only  
python run_pipeline.py --mode evaluation      # Evaluation only
```

### **View MLflow Experiments:**
```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000 in your browser
```

---

## 🎯 Capstone Project: Build Your Pipeline

### **Your Mission:**
Apply the pipeline architecture to a **classification problem of your choice**, inspired by the Spaceship Titanic example.

### **Success Criteria:**
- ✅ Implement modular pipeline structure
- ✅ Advanced feature engineering for your domain
- ✅ Multi-model comparison and selection
- ✅ MLflow experiment tracking
- ✅ Streamlit deployment interface
- ✅ Professional documentation

### **Project Ideas (Inspired by Spaceship Titanic):**
- 🏥 **Medical Diagnosis** - Patient outcome prediction
- 💳 **Fraud Detection** - Transaction classification  
- 📧 **Spam Filter** - Email categorization system
- 🛒 **Customer Churn** - Retention prediction
- 🎯 **Sentiment Analysis** - Review classification
- 🚀 **Custom Dataset** - Your own classification problem!

### **Follow the Pattern:**
Study the `Pipeline/` structure and adapt it for your project:
- Replace dataset loading in `data/load_data.py`
- Customize feature engineering in `data/feature_engineering.py`
- Modify model portfolio in `models/base_model.py`
- Update the Streamlit app for your domain

---

## 🌟 Student Inspiration: Spaceship Titanic Pipeline

The included `Pipeline/` demonstrates **exactly what you'll build**:

### **What Makes It Professional:**
- **Configuration Management**: Centralized settings in `config.py`
- **Feature Engineering**: Custom `SpaceshipFeatureEngineer` class
- **Model Portfolio**: 7+ algorithms with hyperparameter tuning
- **Experiment Tracking**: MLflow for reproducibility
- **Modular Design**: Each component independently testable

### **Key Learning Outcomes:**
After studying this pipeline, you'll be able to:
✅ Build modular ML pipelines from scratch  
✅ Implement domain-specific feature engineering  
✅ Compare multiple models systematically  
✅ Track experiments with MLflow  
✅ Create reproducible research  
✅ Structure projects for collaboration  

### **Adaptation Guide:**
```python
# In your project, replace Spaceship Titanic specifics:
# data/load_data.py → Your dataset loading
# data/feature_engineering.py → Your domain features  
# models/base_model.py → Your model portfolio
# streamlit_app/app.py → Your application interface
```

---

## 📊 Reading MLflow Results

When you run the pipeline, MLflow logs every experiment. Here's how to interpret the results:

```bash
# Open the MLflow dashboard
mlflow ui --backend-store-uri Pipeline/mlruns
# Then open: http://localhost:5000
```

| Metric | What It Means | Good Range |
|--------|--------------|------------|
| `val_accuracy` | Accuracy on held-out validation set | >85% for most problems |
| `val_f1` | Harmonic mean of precision and recall | >0.80 for balanced classes |
| `val_roc_auc` | Area under the ROC curve | >0.85 is strong |
| `train_accuracy` vs `val_accuracy` | Gap > 10% = overfitting | Keep gap under 5% |

**Model selection rule:** pick the model with the best `val_f1`, not `train_accuracy`. High training accuracy with low val accuracy = overfitting — try more regularization or less complexity.

---

## 🔧 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `mlflow.exceptions.MlflowException` | Tracking dir not found | Run pipeline from the `Pipeline/` directory |
| Streamlit app shows stale predictions | Old model pickle loaded | Delete `models/` and re-run `--mode full` |
| `KeyError` in feature engineering | Column renamed in your dataset | Update column names in `config.py` |
| Hyperparameter tuning takes too long | Too many combinations | Reduce `param_grid` in `models/base_model.py` |
| `Class imbalance` warning | Unequal class distribution | Add `class_weight='balanced'` to your estimator |

---

## 🤝 Get Help & Connect

Building pipelines can be challenging - we're here to help!

[![Telegram](https://img.shields.io/badge/Telegram-Join_SAIR_Community-blue?logo=telegram)](https://t.me/+jPPlO6ZFDbtlYzU0)

Get architecture reviews, pipeline feedback, and join deep-dive sessions on ML engineering best practices. Share your pipeline adaptations and get inspired by others!

---

## 🎯 Ready for Your Next Step?

### **Starting classification?**
→ Begin with [`Lecture_4.ipynb`](Lecture_4.ipynb)

### **Ready for pipelines?**
→ Study the [`Pipeline/`](Pipeline/) example thoroughly

### **Want to test the complete system?**
→ Run [`breast_cancer_pipline.py`](breast_cancer_pipline.py)

### **Ready to build your own?**
→ Create your project following the pipeline pattern

### **Ready to advance?**
→ Continue to [Module 3: Neural Networks from Scratch](../3_Neural%20Network%20from%20scratch/README.md)

---

## 📚 Reference Materials

| Resource | Purpose | When to Use |
|----------|---------|-------------|
| [`Pipeline/run_pipeline.py`](Pipeline/run_pipeline.py) | Complete pipeline example | Learning architecture |
| [`Pipeline/streamlit_app/app.py`](Pipeline/streamlit_app/app.py) | Production deployment | Building your UI |
| [`Pipeline/config/config.py`](Pipeline/config/config.py) | Configuration template | Project setup |
| [`breast_cancer_pipline.py`](breast_cancer_pipline.py) | Integrated example | Testing end-to-end flow |

---

> **"السير" - "Walking on a road"**  
> *Professional ML is about systems, not just models. This pipeline example shows you the path from notebooks to production.*

**Study the pattern, then build your masterpiece! 🏗️**

---

**🔜 Next Step:** [Module 3: Neural Networks from Scratch](../3_Neural%20Network%20from%20scratch/README.md)

---

## 🗂️ **Module Structure:**
```
3_Classification/
│
├── 📚 README.md                          # This guide
├── 🎯 Lecture_4.ipynb                    # Classification from Scratch
├── 🚀 Lecture_5.ipynb                    # Production Pipeline Design
├── 🔧 breast_cancer_pipline.py           # Integrated Example
└── 🏗️ Pipeline/                         # Professional Architecture
    ├── config/                           # Configuration Management
    ├── data/                             # Data Processing
    ├── models/                           # ML Modeling
    ├── utils/                            # Shared Utilities
    ├── streamlit_app/                    # Deployment Interface
    ├── run_pipeline.py                   # Main Execution
    ├── README.md                         # Detailed Documentation
    └── requirements.txt                  # Dependencies
```
