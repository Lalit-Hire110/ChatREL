# ChatREL — WhatsApp Relationship Analyzer (v4)

A research-grade prototype that analyzes WhatsApp-style chats to infer **relationship types** and **relationship health** using behavioral features (timing, emoji usage, reply patterns) combined with NLP models.

This repository contains the ChatREL v4 codebase, including data pipelines, model code, demo web UI, and integrations with Hugging Face models and **Contextual Semantic Memory (CSM)**.

> ⚠️ **Security note:**  
> Do **NOT** commit secrets (API keys, tokens, `.env`) to GitHub.  
> Use environment variables or a local `.env` file (added to `.gitignore`).

---

## Table of Contents
- [Project Report](#project-report)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Quickstart (Setup & Run)](#quickstart-setup--run)
- [Environment Variables](#environment-variables)
- [Running the Demo Locally](#running-the-demo-locally)
- [Data & Privacy](#data--privacy)
- [Testing & Evaluation](#testing--evaluation)
- [Contributing](#contributing)
- [Credits & Acknowledgements](#credits--acknowledgements)

---

## Project Report

Download the full project report (detailed methodology, dataset descriptions, experiments, and metrics):

🔗 **[ChatREL — Project Report (Google Drive)](https://drive.google.com/file/d/1QVL3ybM1m_khcUSXYU1CIXkjYF2NkDOb/view?usp=sharing)**

---

## Key Features

- Window-level relationship classification:
  - Couple
  - Crush
  - Friends
  - Family
- Behavioral feature extraction:
  - Message intervals
  - Reply latency
  - Emoji density
  - Message length distributions
- NLP integrations:
  - Sentiment and emotion scoring
  - Hugging Face model calls (optional)
- **Contextual Semantic Memory (CSM)**:
  - Caches HF model scores for repeated phrases
  - Reduces API calls and latency
- Lightweight web demo UI:
  - HTML templates and static assets
- Synthetic chat data generator:
  - Enables class-balanced offline training and testing

---

## Repository Structure (High Level)

This is a representative layout. Actual files may vary.

```text
ChatREL/
├── app/ or web/                # Web demo (templates, static, run scripts)
├── core/                       # Feature extraction, preprocessing, utilities
├── models/                     # Model training, inference, saved artifacts
├── data/                       # Sample inputs / synthetic data (no real chats)
├── notebooks/                  # EDA & experiments
├── requirements.txt
├── .gitignore
├── .env.example
└── verify_config.py

Tech Stack
Python 3.9+

Data & ML: pandas, numpy, scikit-learn, xgboost

NLP: Hugging Face Transformers / Inference API (optional)

Web Demo: Flask (demo web UI)

Alternative Demo: Streamlit (optional)

Explainability: SHAP (optional)

Requirements
Python 3.9 or newer

Create virtual environment & install dependencies:

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

Environment Variables
Use environment variables for all secrets and configurable settings.

Common variables:

HF_API_KEY — Hugging Face API key

FLASK_ENV — development / production (optional)

CSM_CACHE_PATH — Path for CSM cache (optional)

Running the Demo Locally
The project includes a simple web demo (Flask-style templates + static assets).
Exact entrypoint may vary (run.py, web/app.py, app.py).

Using Flask
# Unix / macOS
export FLASK_APP=web
export FLASK_ENV=development
flask run

# Windows PowerShell
$env:FLASK_APP = "web"
$env:FLASK_ENV = "development"
flask run

If run.py exists
python run.py

If unsure, look for:

if __name__ == "__main__":

Files named run.py or app.py

Data & Privacy
Do NOT use real private chat exports without explicit consent.

Remove or anonymize:

Names

Phone numbers

Media

Repository includes synthetic or redacted data only.

Testing & Evaluation
Unit tests (if present) are located in tests/

Run tests using:
pytest

Evaluation metrics (see project report):

Classification: Precision, Recall, F1-score

Regression (if applicable): RMSE, R²

Contributing
Open an issue before major changes

Fork → feature branch → PR

Keep secrets out of commits

Use .env.example for reproducibility

Credits & Acknowledgements
Built as part of academic and hackathon work (Smart India Hackathon / SAFAL context).

ChatREL evolved from v3 → v4, introducing Contextual Semantic Memory (CSM) and Hugging Face model integration to improve interpretability and reduce API overhead.
