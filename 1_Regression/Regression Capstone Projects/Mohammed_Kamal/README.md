# ⛽ Albacete Gas Station Intelligence System

## Developed by: Mohammed Kamal Part of the SAIR ML/DL Bootcamp


<!-- ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg) -->

## 🎯 Project Overview
This project is an end-to-end Machine Learning solution designed to predict and analyze gasoline prices (Gasolina 95) in **Albacete, Spain**. By integrating geographic, brand, and temporal data, the system provides accurate price estimations and market insights for both consumers and businesses.

## 📊 Dataset Description
The project utilizes two integrated datasets:
- **`gasolineras_ab.csv`**: Contains static information about gas stations (Latitude, Longitude, Brand Name, and Location).
- **`precios_gasolineras.csv`**: Contains time-series price data for various fuel types across different dates.

## 🧠 Advanced Machine Learning Workflow

### 1. Feature Engineering (The Game Changer) 🚀
To move beyond basic prediction, we implemented **Temporal Feature Engineering**:
- Converted raw date strings into `DateTime` objects.
- Extracted the **'Month'** as a numerical feature. 
- **Impact:** This allowed the model to understand seasonality and price fluctuations over time, significantly improving accuracy.

### 2. Production-Grade Pipeline
Used a Scikit-Learn `Pipeline` to ensure a robust and leak-free workflow:
- **Numerical Scaling**: Used `StandardScaler` for coordinates and month data.
- **Categorical Encoding**: Used `OneHotEncoder` for station brands (`rotulo`).
- **Model**: Switched to **Gradient Boosting Regressor** for superior performance in capturing non-linear relationships.

### 3. Experiment Tracking with MLflow
All model versions and parameters are tracked using **MLflow**:
- Logged metrics: **R² Score** and **RMSE**.
- Saved model artifacts in a specialized `production_models/` directory for easy deployment.
- Enabled comparison between base models and feature-engineered models.

### Run the Analysis & Train the model:
- Open Gas_Price_Analysis.ipynb and run all cells. This will generate the mlflow.and the saved model.

## 💻 Interactive Dashboard (Streamlit)
The deployment side features a dual-tab interface:
- **Market Analysis**: Interactive maps and brand-wise price comparisons using `Plotly`.
- **Price Predictor**: A real-time tool where users input location and brand to get an instant AI-driven price estimate based on the current month.


## Launch the App:
- streamlit run app.py

## View MLflow Dashboard:
- mlflow ui --backend-store-uri sqlite:///mlflow.db

## 🛠️ Installation & Usage

1. **Clone the project** and navigate to the directory.
2. **Set up the environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt