#%% Tabular Boosting Models
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import os

# 80% train, 20% test split

def run_model(Classifier, name, X_train, y_train, X_test, y_test):
    print(f"Training {name}...")
    model = Classifier()
    model.fit(X_train, y_train)
    print(f"Predicting with {name}...")
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    return report

def run_boosting_models(csv_path):
    print("Loading and preprocessing data...")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    df, features = load_and_preprocess(csv_path)
    print(f"HALO unique classes: {df['HALO'].unique()}")
    print("Using 80% train, 20% test split (random)")
    print("Splitting CME data...")
    
    results = []
    
    # CME as target
    features_cme = [f for f in features if f != 'HALO']
    X_cme = df[features_cme]
    y_cme = df['CME']
    X_train_cme, X_test_cme, y_train_cme, y_test_cme = train_test_split(X_cme, y_cme, test_size=0.2, random_state=42, stratify=y_cme)
    print("\n--- CME Classification ---")
    for clf, name in [
        (LGBMClassifier, "LightGBM"),
        (XGBClassifier, "XGBoost"),
        (lambda: CatBoostClassifier(verbose=0), "CatBoost")
    ]:
        report = run_model(clf, f"{name} - CME", X_train_cme, y_train_cme, X_test_cme, y_test_cme)
        results.append(f"=== {name} - CME ===")
        results.append(report)
        results.append("")
    
    results.append(f"CME unique classes: {df['CME'].unique()}")
    results.append("CME class distribution:")
    results.append(str(df['CME'].value_counts()))
    results.append("")
    
    print("Splitting HALO data...")
    # HALO as target
    features_halo = [f for f in features if f != 'CME']
    X_halo = df[features_halo]
    y_halo = df['HALO']
    X_train_halo, X_test_halo, y_train_halo, y_test_halo = train_test_split(X_halo, y_halo, test_size=0.2, random_state=42, stratify=y_halo)
    print("\n--- HALO Classification ---")
    for clf, name in [
        (LGBMClassifier, "LightGBM"),
        (XGBClassifier, "XGBoost"),
        (lambda: CatBoostClassifier(verbose=0), "CatBoost")
    ]:
        report = run_model(clf, f"{name} - HALO", X_train_halo, y_train_halo, X_test_halo, y_test_halo)
        results.append(f"=== {name} - HALO ===")
        results.append(report)
        results.append("")
    
    results.append(f"Total samples: {len(df)}")
    results.append("HALO class distribution:")
    results.append(str(df['HALO'].value_counts()))
    
    # Save results
    with open("results/boosting_models_results.txt", "w") as f:
        f.write("\n".join(results))
    
    print(f"\nTotal samples: {len(df)}")
    print("Detailed results saved to results/boosting_models_results.txt")
