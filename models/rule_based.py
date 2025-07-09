#%% Rule-Based Benchmark
from sklearn.metrics import f1_score, classification_report
from utils.preprocessing import load_and_preprocess
import os

def rule_based(csv_path):
    print("Running rule-based benchmark...")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    df, features = load_and_preprocess(csv_path)
    split_time = df['epoch_for_cdf_mod'].quantile(0.8)
    test = df[df['epoch_for_cdf_mod'] >= split_time]
    print(f"Test set size: {len(test)}")
    
    results = []
    
    # CME rule
    rule_cme = ((test['proton_density'] > 5) & (test['proton_bulk_speed'] > 400)).astype(int)
    score_cme = f1_score(test['CME'], rule_cme)
    cme_report = classification_report(test['CME'], rule_cme, zero_division=0)
    
    results.append("=== Rule-Based CME Classification ===")
    results.append(f"CME F1 Score: {score_cme:.4f}")
    results.append(f"Classification Report:")
    results.append(cme_report)
    results.append("")
    
    # HALO rule (example: high speed and high density -> class 1, else 0)
    print("\nRule-based HALO classification...")
    rule_halo = ((test['proton_density'] > 5) & (test['proton_bulk_speed'] > 400)).astype(int)
    halo_f1 = f1_score(test['HALO'], rule_halo, average='weighted')
    halo_report = classification_report(test['HALO'], rule_halo, zero_division=0)
    
    results.append("=== Rule-Based HALO Classification ===")
    results.append(f"HALO unique classes: {test['HALO'].unique()}")
    results.append(f"HALO class distribution:")
    results.append(str(test['HALO'].value_counts()))
    results.append(f"HALO F1 Score (weighted): {halo_f1:.4f}")
    results.append(f"Classification Report:")
    results.append(halo_report)
    
    # Save results
    with open("results/rule_based_results.txt", "w") as f:
        f.write("\n".join(results))
    
    print(f"Rule-based CME F1: {score_cme:.4f}")
    print(f"Rule-based HALO F1: {halo_f1:.4f}")
    print("Detailed results saved to results/rule_based_results.txt")
