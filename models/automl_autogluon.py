#%% AutoML with AutoGluon
from autogluon.tabular import TabularPredictor
from utils.preprocessing import load_and_preprocess
from sklearn.metrics import f1_score, classification_report
import os
import shutil

def run_autogluon(csv_path):
    print("Running AutoGluon...")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Clean up any existing model directories
    model_dirs = ["autogluon_models", "AutogluonModels"]
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"Removing existing model directory: {model_dir}")
            shutil.rmtree(model_dir)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df, features = load_and_preprocess(csv_path)
    # Split data
    split_time = df['epoch_for_cdf_mod'].quantile(0.8)
    train = df[df['epoch_for_cdf_mod'] < split_time]
    test = df[df['epoch_for_cdf_mod'] >= split_time]
    print(f"Train set size: {len(train)}, Test set size: {len(test)}")
    
    # Check class distributions
    print(f"CME distribution in train: {train['CME'].value_counts().to_dict()}")
    print(f"HALO distribution in train: {train['HALO'].value_counts().to_dict()}")
    
    results = []
    
    # CME Model Training
    print("\n" + "="*50)
    print("TRAINING CME MODEL")
    print("="*50)
    
    cme_model_path = "autogluon_models/cme_predictor"
    print("Training CME classifier...")
    
    try:
        predictor_cme = TabularPredictor(label='CME', path=cme_model_path).fit(
            train[features + ['CME']], 
            presets='best_quality',
            ag_args_fit={'num_gpus': 0, 'num_cpus': 8},
            hyperparameters={
                'GBM': {'ag_args_fit': {'num_cpus': 8}},
                'CAT': {'ag_args_fit': {'num_cpus': 8}},
                'RF': {'ag_args_fit': {'num_cpus': 8}},
                'NN_TORCH': {'ag_args_fit': {'num_cpus': 8}}
            }
        )
        
        # CME predictions and evaluation
        print("Making CME predictions...")
        cme_pred = predictor_cme.predict(test[features])
        cme_f1 = f1_score(test['CME'], cme_pred, average='weighted')
        print(f"CME F1-score: {cme_f1:.4f}")
        
        # Save CME model leaderboard
        try:
            leaderboard = predictor_cme.leaderboard(test[features + ['CME']], silent=True)
            print("\nCME Model Leaderboard:")
            print(leaderboard.head(10))
        except Exception as e:
            print(f"Could not display CME leaderboard: {e}")
        
        results.append({
            'Model': 'AutoGluon',
            'Target': 'CME',
            'F1_Score': cme_f1,
            'Classification_Report': classification_report(test['CME'], cme_pred)
        })
        
    except Exception as e:
        print(f"Error training CME model: {e}")
        results.append({
            'Model': 'AutoGluon',
            'Target': 'CME',
            'F1_Score': 0.0,
            'Classification_Report': f"Error: {str(e)}"
        })
    
    # HALO Model Training
    print("\n" + "="*50)
    print("TRAINING HALO MODEL")
    print("="*50)
    
    halo_model_path = "autogluon_models/halo_predictor"
    print("Training HALO classifier...")
    
    try:
        predictor_halo = TabularPredictor(label='HALO', path=halo_model_path).fit(
            train[features + ['HALO']], 
            presets='best_quality',
            ag_args_fit={'num_gpus': 0, 'num_cpus': 8},
            hyperparameters={
                'GBM': {'ag_args_fit': {'num_cpus': 8}},
                'CAT': {'ag_args_fit': {'num_cpus': 8}},
                'RF': {'ag_args_fit': {'num_cpus': 8}},
                'NN_TORCH': {'ag_args_fit': {'num_cpus': 8}}
            }
        )
        
        # HALO predictions and evaluation
        print("Making HALO predictions...")
        halo_pred = predictor_halo.predict(test[features])
        halo_f1 = f1_score(test['HALO'], halo_pred, average='weighted')
        print(f"HALO F1-score: {halo_f1:.4f}")
        
        # Save HALO model leaderboard
        try:
            leaderboard = predictor_halo.leaderboard(test[features + ['HALO']], silent=True)
            print("\nHALO Model Leaderboard:")
            print(leaderboard.head(10))
        except Exception as e:
            print(f"Could not display HALO leaderboard: {e}")
        
        results.append({
            'Model': 'AutoGluon',
            'Target': 'HALO',
            'F1_Score': halo_f1,
            'Classification_Report': classification_report(test['HALO'], halo_pred)
        })
        
    except Exception as e:
        print(f"Error training HALO model: {e}")
        results.append({
            'Model': 'AutoGluon',
            'Target': 'HALO',
            'F1_Score': 0.0,
            'Classification_Report': f"Error: {str(e)}"
        })
    
    # Save detailed results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    print("Saving AutoGluon results...")
    with open("results/autogluon_results.txt", "w") as f:
        f.write("AutoGluon Results\n")
        f.write("="*50 + "\n\n")
        
        for result in results:
            f.write(f"Model: {result['Model']}\n")
            f.write(f"Target: {result['Target']}\n")
            f.write(f"F1 Score: {result['F1_Score']:.4f}\n")
            f.write(f"Classification Report:\n{result['Classification_Report']}\n")
            f.write("-" * 50 + "\n\n")
    
    print("AutoGluon results saved to results/autogluon_results.txt")
    
    # Print summary
    print("\nAutoGluon Summary:")
    for result in results:
        print(f"  {result['Target']}: F1 = {result['F1_Score']:.4f}")
    
    return results

if __name__ == "__main__":
    results = run_autogluon("merged_data.csv")
    print("\nAutoGluon Pipeline Complete!")
    for result in results:
        print(f"{result['Target']}: {result['F1_Score']:.4f}")