# 📈 Bitcoin Cycle Forecaster: An MLOps Approach

This project is a production-grade Machine Learning pipeline designed to analyze and forecast Bitcoin (BTC) price cycles. Instead of a simple research script, it implements **MLOps best practices**, modular software engineering, and experiment tracking to ensure scalability and reproducibility.

---

## 🎯 Project Objective
The goal is to provide a robust framework for forecasting BTC prices for the next halving cycle (4 years) using the **Prophet** architecture while managing the entire ML lifecycle with **MLflow**.

## 🛠 Tech Stack
*   **Language:** Python 3.10+
*   **Forecasting:** [Prophet](https://facebook.github.io/prophet/) (Meta's time-series model)
*   **MLOps & Tracking:** [MLflow](https://mlflow.org/)
*   **Data Ingestion:** Yahoo Finance API (`yfinance`)
*   **Configuration:** YAML
*   **Environment Management:** Virtualenv / Requirements.txt

## 🏗 Project Structure
The project follows a modular architecture, separating data, configuration, and source code—a key requirement for production-stable AI systems.

```text
BITCOIN-CYCLE-FORECASTER/
├── config/               # Centralized configuration (YAML)
│   └── model_config.yaml # Model hyperparameters and data tickers
├── data/                 # Data versioning directory
│   ├── raw/              # Historical data (tracked by DVC/Local)
│   └── processed/        # Model outputs and forecasts
├── models/               # Locally saved model artifacts and plots
├── src/                  # Source code (Modular Engineering)
│   ├── download_data.py  # Data Ingestion Pipeline
│   └── train.py          # Training, Evaluation & MLflow Logging
├── venv/                 # Isolated virtual environment
├── requirements.txt      # Dependency management
└── README.md             # Project documentation
```

## 🚀 Getting Started

### 1. Installation & Environment Setup
Ensure you are using an isolated environment to avoid dependency conflicts.

```bash
# Clone the repository
git clone https://github.com/BuberDev/bitcoin-cycle-forecaster.git
cd bitcoin-cycle-forecaster

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Ingestion
Download the latest historical BTC data based on the configuration file.
```bash
python src/download_data.py
```

### 3. Model Training & Experiment Tracking
Train the Prophet model and log all parameters, metrics, and plots to MLflow.
```bash
python src/train.py
```

---

## 📊 MLOps: Experiment Tracking with MLflow
This project utilizes **MLflow** to manage the model lifecycle. Every training run logs:
*   **Parameters:** `changepoint_prior_scale`, `seasonality_mode`, `ticker`.
*   **Metrics:** `MAE` (Mean Absolute Error), `RMSE`.
*   **Artifacts:** Forecast plots (.png), predicted data (.csv), and the serialized model object.

**To view the dashboard:**
```bash
mlflow ui
```
Then navigate to `http://127.0.0.1:5000` in your browser.

---
## 🐳 Docker Support
The project is fully containerized for production consistency.

**Build the image:**
```bash
docker build -t btc-forecaster .

------

## 💡 Key Engineering Features Implemented
*   **Decoupled Configuration:** No hardcoded variables. All model parameters are managed via `config/model_config.yaml`.
*   **Resource Management:** Explicit use of `matplotlib` figure objects and memory cleanup (`plt.close(fig)`) for scalable batch processing.
*   **Automated Pipeline:** Modular scripts allow for easy integration into CI/CD pipelines (GitHub Actions/Jenkins).
*   **Error Handling:** Robust path checking and directory creation for automated environments.

## 🏁 Future Roadmap
- [ ] **AI Agent Integration:** Add an LLM-based agent to provide natural language insights on the generated forecasts.
- [ ] **Real-time Predictions:** Implement a REST API using FastAPI for real-time inference.

---
**Author:** Dawid Bubernak  
**Role:** Machine Learning Engineer Candidate

---