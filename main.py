from models.lightgbm_xgboost_catboost import run_boosting_models
from models.lightgbm_xgboost_catboost_smote import run_boosting_models_smote
from models.rule_based import rule_based
from models.automl_autogluon import run_autogluon
from compare_results import compare_model_results
import os

csv_path = "data/merged_data.csv"

print("Script started")
print("Results will be saved to 'results/' directory for easy comparison")

# Create results directory
os.makedirs("results", exist_ok=True)

try:
        # print("\n=== Boosting Models (No SMOTE) ===")
        # run_boosting_models(csv_path) 
        # print("\n=== Boosting Models (With SMOTE) ===")
        # run_boosting_models_smote(csv_path)
    # print("\n=== Rule-Based Model ===")
    # rule_based(csv_path)
    print("\n=== AutoML (AutoGluon) ===")
    run_autogluon(csv_path)
    print("\nAll models finished successfully.")
    
    print("\n=== Generating Comparison Summary ===")
    compare_model_results()
    
except Exception as e:
    print("Error:", e)

print("\n" + "="*60)
print("CHECK THESE FILES FOR DETAILED RESULTS:")
print("="*60)
print("📄 results/boosting_models_results.txt - LightGBM, XGBoost, CatBoost")
print("📄 results/catboost_smote_results.txt - CatBoost with/without SMOTE")
print("📄 results/rule_based_results.txt - Rule-based benchmark")
print("📄 results/autogluon_results.txt - AutoML results")
print("📄 results/comparison_summary.txt - Quick comparison overview")
print("="*60)
# add more models...
