# CME & HALO Prediction Pipeline

This repository contains a robust, reproducible machine learning pipeline for predicting CME and HALO events using tabular data. The pipeline supports boosting models (CatBoost, LightGBM, XGBoost), deep learning (LSTM), rule-based baselines, and AutoML (AutoGluon). All results are saved for easy comparison.

## 🏆 **Final Model Recommendation: CatBoost**

After extensive evaluation of multiple approaches, **CatBoost** is our recommended final choice for production deployment:

- **Robust Performance:** Consistently strong results across CME and HALO prediction tasks
- **Production Ready:** Stable, well-tested, and handles categorical features natively
- **Balanced Complexity:** Good performance without excessive overfitting risk
- **Interpretable:** Feature importance and model behavior can be easily analyzed

### AutoGluon Consideration
While AutoGluon achieved the highest validation scores in our experiments, we recommend CatBoost over AutoGluon for the following reasons:
- **Overfitting Risk:** AutoGluon's ensemble approach and extensive hyperparameter search may lead to overfitting on this specific dataset
- **Complexity:** AutoGluon's black-box nature makes it harder to interpret and debug
- **Production Stability:** CatBoost offers more predictable behavior in production environments
- **Resource Efficiency:** CatBoost requires fewer computational resources for training and inference

## Features
- **Tabular ML Models:** **CatBoost (Final Choice)**, LightGBM, XGBoost
- **Deep Learning:** LSTM with comprehensive preprocessing
- **Rule-Based Baseline**
- **AutoML:** AutoGluon (research/comparison purposes)
- **Ensemble Models:** RandomForest + ExtraTrees + MLP + LightGBM combinations
- **Handles Imbalanced Data:** SMOTE support and class balancing
- **Reproducible:** Consistent preprocessing and feature engineering
- **Easy Comparison:** All results saved to `results/` directory
- **Progress Tracking:** Comprehensive progress bars for large datasets
- **Colab Ready:** Minimal script for quick experimentation

## Directory Structure
```
code/
├── main.py                       # Main pipeline runner
├── autogluon_colab.py           # Minimal AutoGluon pipeline for Colab
├── ensemble_model.py            # Complete ensemble pipeline (both CME & HALO)
├── ensemble_cme.py              # Optimized CME-only ensemble model
├── ensemble_halo.py             # Optimized HALO-only ensemble model
├── run_model.py                 # Simple script to run pickle models
├── models/                      # Individual model scripts
│   ├── lightgbm_xgboost_catboost.py      # CatBoost (Final Choice)
│   ├── lightgbm_xgboost_catboost_smote.py # CatBoost with SMOTE
│   ├── lstm_model.py            # LSTM deep learning model
│   ├── rule_based.py            # Rule-based baseline
│   └── automl_autogluon.py      # AutoGluon (research use)
├── utils/
│   └── preprocessing.py         # Data loading and feature engineering
├── data/                        # Input data (e.g., merged_data.csv)
├── results/                     # All output and result files
├── requirements.txt             # Python dependencies
├── test_ensemble_progress.py    # Test script for progress bars
└── .gitignore                   # Ignore patterns for code/data/models
```

## .gitignore (Ignored Files & Folders)
The following are ignored by default (see `.gitignore`):
- Python virtual environments: `venv/`, `.env/`, `.venv/`
- Python cache: `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd`
- AutoGluon trained models: `autogluon_models/`
- LSTM model: `best_lstm_model.h5`
- CatBoost info: `catboost_info/`
- Jupyter checkpoints: `.ipynb_checkpoints/`
- IDE/editor settings: `.vscode/`, `.idea/`
- OS files: `.DS_Store`, `Thumbs.db`
- Results, outputs, and logs: `results/`, `outputs/`, `logs/`
- Temporary files: `*.tmp`, `*.temp`, `*.log`
- Data files (optional): `data/`, `datasets/`, `*.csv`, `*.parquet`, `*.json`

To track any of these, comment out the relevant line in `.gitignore`.

## Quick Start

### 🎯 **Recommended Workflow (CatBoost)**
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd code
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Prepare your data:**
   - Place your CSV file in the root directory (e.g., `merged_data.csv`)
4. **Run CatBoost models (recommended):**
   ```sh
   python main.py
   ```
   This will run all models including our recommended CatBoost approach.

### 🚀 **Alternative: Run Individual Models**
- **CatBoost only:** `python models/lightgbm_xgboost_catboost.py`
- **Ensemble models:** `python ensemble_cme.py` or `python ensemble_halo.py`
- **LSTM:** `python models/lstm_model.py`
- **AutoGluon (research):** `python models/automl_autogluon.py`

### 📊 **Run Pickle Model (from Downloads)**
If you have a trained AutoGluon pickle model:
```sh
python run_model.py
```
This script will automatically find pickle files in common locations including your Downloads folder.

### 🧪 **Colab Experimentation**
For quick experimentation in Google Colab:
- Upload `autogluon_colab.py` and your data to Colab
- Run: `from autogluon_colab import run_autogluon; run_autogluon('merged_data.csv')`

## Results & Model Comparison

### 📁 **Output Files**
All results are automatically saved to the `results/` directory:
- **`results/catboost_results.txt`** - CatBoost performance (our recommendation)
- **`results/autogluon_results.txt`** - AutoGluon output (research reference)
- **`results/ensemble_cme_results.txt`** - CME ensemble results
- **`results/ensemble_halo_results.txt`** - HALO ensemble results
- **`results/comparison_summary.txt`** - Quick comparison overview
- **`results/plots/`** - Confusion matrices and ROC curves

### 🏅 **Model Selection Rationale**
| Model | Performance | Overfitting Risk | Production Readiness | Recommendation |
|-------|-------------|------------------|---------------------|----------------|
| **CatBoost** | High | Low | ✅ High | **🏆 Final Choice** |
| AutoGluon | Highest | ⚠️ High | Medium | Research Only |
| LightGBM | High | Medium | High | Good Alternative |
| XGBoost | High | Medium | High | Good Alternative |
| Ensemble | High | Medium | Medium | Experimental |
| LSTM | Medium | High | Medium | Specialized Use |

### 📊 **Performance Expectations**
Based on our experiments:
- **CatBoost:** Balanced performance with robust generalization
- **AutoGluon:** Highest validation scores but potential overfitting
- **Ensemble Models:** Good performance with comprehensive progress tracking
- **LSTM:** Suitable for temporal pattern detection

## Notes & Best Practices

### 🎯 **Production Deployment**
- **Use CatBoost** for production deployments due to stability and interpretability
- Monitor for data drift and retrain periodically
- Validate model performance on unseen data before deployment

### 🔬 **Research & Development**
- AutoGluon can be used for research purposes and upper-bound performance estimation
- Ensemble models provide good experimental baselines
- LSTM models are useful for investigating temporal dependencies

### 📂 **File Management**
- Large model files and data are ignored by default via `.gitignore`
- To track model files, comment out the relevant lines in `.gitignore`
- All results are automatically saved with timestamps for comparison

### ⚡ **Performance Optimization**
- Ensemble models include progress bars for large datasets (4M+ rows)
- Data is processed in chunks for memory efficiency
- Models are optimized for speed while maintaining accuracy

### 🔧 **Reproducibility**
- Use the provided preprocessing utilities for consistent results
- All models use fixed random seeds where applicable
- Requirements.txt includes all necessary dependencies with version constraints

## License
MIT License

---

**Contact:** For questions or contributions, open an issue or pull request on GitHub.

