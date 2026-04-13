import os

readme_content = """
# 🏥 Medical Insurance Cost Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)

## 🎯 Project Overview
This project aims to predict medical insurance costs based on individual patient data such as age, BMI, smoking status, and region. It utilizes a robust Machine Learning pipeline to provide accurate financial estimates, helping insurance providers or individuals understand the factors driving healthcare costs.

## 📊 Dataset Description
The dataset contains information on 1,338 patients:
- **Age**: Age of primary beneficiary.
- **Sex**: Insurance contractor gender (female, male).
- **BMI**: Body Mass Index, providing an understanding of body weight relative to height.
- **Children**: Number of children covered by health insurance.
- **Smoker**: Smoking status (yes, no).
- **Region**: The beneficiary's residential area in the US.
- **Charges (Target)**: Individual medical costs billed by health insurance.

## 🧠 Machine Learning Workflow

### 1. Data Preprocessing & EDA
- Conducted Exploratory Data Analysis (EDA) to find correlations.
- Discovered that **Smoking status** is the most significant predictor of cost.
- Handled categorical variables using **OneHotEncoding** and numerical scaling via **StandardScaler**.

### 2. Model Selection (The Battle of Models)
We compared several regression algorithms using a unified Pipeline:
| Model | R² Score |
| :--- | :--- |
| **Gradient Boosting** | **0.8787** 🏆 |
| Random Forest | 0.8754 |
| SVR (Support Vector) | 0.8496 |
| Linear Regression | 0.7836 |

### 3. Professional Pipeline
The final model is built into a **Scikit-Learn Pipeline** which bundles the preprocessing and the regressor together. This ensures that the data is transformed identically during both training and real-time prediction.

## 💻 Deployment
The project includes a web-based interface built with **Streamlit**.
- **User Friendly**: Input fields for age, weight, and habits.
- **Instant Prediction**: The "Champion Model" (Gradient Boosting) provides a cost estimate in real-time.

## 🛠️ How to Run
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/medical-insurance-cost.git](https://github.com/Mohammed_Kamal/medical-insurance-cost.git)

2. **Install dependencies**:
    ```bash
    uv sync

3. **Install dependencies**:
    ```bash
    streamlit run app.py
