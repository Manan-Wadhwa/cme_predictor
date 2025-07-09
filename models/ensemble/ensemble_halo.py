# Fast HALO Ensemble Prediction Model
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class HALOEnsemblePredictor:
    def __init__(self):
        # Define ensemble weights as specified
        self.ensemble_weights = {
            'RandomForestEntr': 0.333,
            'ExtraTreesEntr': 0.333,
            'NeuralNetTorch': 0.167,
            'LightGBMLarge': 0.167
        }
        
        # Initialize ultra-fast models for 4M+ dataset
        self.models = {
            'RandomForestEntr': RandomForestClassifier(
                n_estimators=50,   # Drastically reduced
                max_depth=8,       # Very shallow
                min_samples_split=100,  # High for speed
                min_samples_leaf=50,    # High for speed
                max_features='sqrt',    # Faster feature selection
                random_state=42,
                n_jobs=-1,
                bootstrap=True
            ),
            'ExtraTreesEntr': ExtraTreesClassifier(
                n_estimators=50,   # Drastically reduced
                max_depth=8,       # Very shallow
                min_samples_split=100,
                min_samples_leaf=50,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                bootstrap=True
            ),
            'NeuralNetTorch': MLPClassifier(
                hidden_layer_sizes=(64,),  # Single small layer
                activation='relu',
                solver='sgd',      # Faster solver
                alpha=0.01,        # Higher regularization
                learning_rate='constant',
                learning_rate_init=0.1,
                max_iter=100,      # Very low iterations
                random_state=42,
                early_stopping=True,
                validation_fraction=0.05,
                n_iter_no_change=5
            ),
            'LightGBMLarge': lgb.LGBMClassifier(
                n_estimators=100,   # Much smaller
                learning_rate=0.2,  # Higher for faster convergence
                max_depth=6,        # Very shallow
                num_leaves=20,      # Much smaller
                feature_fraction=0.5,  # Sample fewer features
                bagging_fraction=0.5,  # Sample fewer rows
                bagging_freq=1,
                min_data_in_leaf=1000,  # Large for speed
                random_state=42,
                verbose=-1,
                n_jobs=-1,
                boost_from_average=True
            )
        }
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_and_preprocess(self, csv_path, test_on_full=True):
        """Ultra-fast data loading for 4M+ dataset"""
        print(f"🔄 Loading HALO data from {csv_path}...")
        
        # Read in chunks for memory efficiency with progress bar
        chunk_size = 100000
        chunks = []
        
        # First pass to count total rows for progress bar
        print("Counting total rows...")
        total_rows = sum(1 for _ in open(csv_path, 'r')) - 1  # -1 for header
        total_chunks = (total_rows // chunk_size) + 1
        
        print("Reading data in chunks for memory efficiency...")
        chunk_reader = pd.read_csv(csv_path, chunksize=chunk_size, dtype={'CME': 'int8', 'HALO': 'int8'})
        
        for chunk in tqdm(chunk_reader, total=total_chunks, desc="Loading chunks"):
            # Basic cleaning per chunk
            chunk.replace(-1e31, np.nan, inplace=True)
            chunk.dropna(thresh=len(chunk.columns) * 0.3, inplace=True)  # Very aggressive
            chunks.append(chunk)
        
        df_full = pd.concat(chunks, ignore_index=True)
        print(f"Full dataset shape: {df_full.shape}")
        
        # Create training sample and keep full dataset for testing
        df_train = df_full.copy()
        if len(df_full) > 1000000:  # If more than 1M rows, sample for training
            sample_size = min(500000, len(df_full))  # Max 500K rows for training
            print(f"Sampling {sample_size} rows for faster training...")
            df_train = df_full.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Handle datetime efficiently
        for df in [df_train, df_full]:
            if 'epoch_for_cdf_mod' in df.columns:
                df['epoch_for_cdf_mod'] = pd.to_datetime(df['epoch_for_cdf_mod'], cache=True)
        
        # Minimal cleaning with progress
        print("🧹 Cleaning data...")
        for df in tqdm([df_train, df_full], desc="Cleaning datasets"):
            df = df.sort_values('epoch_for_cdf_mod')
            df = df.fillna(method='ffill', limit=3).fillna(0)  # Fast fill
        
        print(f"Training data shape: {df_train.shape}")
        if test_on_full:
            print(f"Testing will be done on full dataset: {df_full.shape}")
        
        # Minimal feature engineering
        for df in [df_train, df_full]:
            if 'epoch_for_cdf_mod' in df.columns:
                df['hour'] = df['epoch_for_cdf_mod'].dt.hour
        
        # Define features (excluding CME since we're predicting HALO)
        columns_to_remove = ['epoch_for_cdf_mod', 'spacecraft_xpos', 'spacecraft_ypos', 'spacecraft_zpos', 'CME']
        features = [col for col in df_train.columns if col not in columns_to_remove + ['HALO']]
        
        # Feature selection for speed - keep only most important
        if len(features) > 10:
            # Keep only numeric features with good variance
            numeric_features = df_train[features].select_dtypes(include=[np.number]).columns
            feature_vars = df_train[numeric_features].var()
            top_features = feature_vars.nlargest(10).index.tolist()
            features = top_features
            print(f"Selected top {len(features)} features for speed")
        
        print(f"HALO Feature columns ({len(features)}): {features}")
        
        return df_train, df_full if test_on_full else df_train, features
    
    def prepare_data(self, df_train, df_test, features):
        """Prepare HALO data for training and testing"""
        print(f"🎯 Preparing HALO data for training and testing...")
        
        # Temporal split for training data
        split_time = df_train['epoch_for_cdf_mod'].quantile(0.8)
        train_subset = df_train[df_train['epoch_for_cdf_mod'] < split_time]
        
        # Use full test dataset
        test_subset = df_test[df_test['epoch_for_cdf_mod'] >= split_time] if 'epoch_for_cdf_mod' in df_test.columns else df_test
        
        X_train = train_subset[features].values  # Convert to numpy for speed
        y_train = train_subset['HALO'].values
        X_test = test_subset[features].values
        y_test = test_subset['HALO'].values
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"HALO distribution in train: {np.bincount(y_train)}")
        print(f"HALO distribution in test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train, y_train):
        """Fast training of HALO ensemble models"""
        print("\n" + "="*50)
        print("🚀 FAST TRAINING HALO ENSEMBLE MODELS")
        print("="*50)
        
        # Scale features once with progress
        print("📊 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        print("✅ Feature scaling completed!")
        
        # Train models with detailed progress tracking
        print(f"\n🎯 Training {len(self.models)} ensemble models...")
        
        # Progress bar for model training
        with tqdm(total=len(self.models), desc="Training models", unit="model") as pbar:
            for model_name, model in self.models.items():
                pbar.set_description(f"Training {model_name}")
                start_time = time.time()
                
                if model_name == 'NeuralNetTorch':
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                pbar.set_postfix({"Time": f"{training_time:.2f}s"})
                pbar.update(1)
                print(f"   ✅ {model_name} completed in {training_time:.2f}s")
        
        self.is_fitted = True
        print("\n🎉 HALO ensemble training completed!")
    
    def predict_proba(self, X):
        """Fast ensemble predictions with probabilities"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction!")
        
        print("🔮 Generating ensemble predictions...")
        X_scaled = self.scaler.transform(X)
        ensemble_proba = np.zeros((len(X), 2))
        
        # Progress bar for ensemble prediction
        with tqdm(total=len(self.models), desc="Ensemble prediction", unit="model") as pbar:
            for model_name, model in self.models.items():
                pbar.set_description(f"Predicting {model_name}")
                weight = self.ensemble_weights[model_name]
                
                if model_name == 'NeuralNetTorch':
                    proba = model.predict_proba(X_scaled)
                else:
                    proba = model.predict_proba(X)
                
                ensemble_proba += weight * proba
                pbar.set_postfix({"Weight": f"{weight:.3f}"})
                pbar.update(1)
        
        print("✅ Ensemble predictions completed!")
        return ensemble_proba
    
    def predict(self, X):
        """Fast ensemble predictions"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def evaluate_comprehensive(self, X_test, y_test):
        """Comprehensive HALO evaluation"""
        print(f"\n" + "="*60)
        print("📊 COMPREHENSIVE HALO EVALUATION")
        print("="*60)
        
        # Get predictions with progress tracking
        print("🔍 Evaluating model performance...")
        with tqdm(total=3, desc="Evaluation steps", unit="step") as pbar:
            # Step 1: Generate predictions
            pbar.set_description("Generating predictions")
            y_pred = self.predict(X_test)
            pbar.update(1)
            
            # Step 2: Generate probabilities
            pbar.set_description("Generating probabilities")
            y_proba = self.predict_proba(X_test)[:, 1]
            pbar.update(1)
            
            # Step 3: Calculate metrics
            pbar.set_description("Calculating metrics")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except:
                roc_auc = 0.0
            
            cm = confusion_matrix(y_test, y_pred)
            pbar.update(1)
        
        # Print results
        print(f"🎯 HALO PERFORMANCE METRICS:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        
        print(f"\n📈 HALO CONFUSION MATRIX:")
        print(cm)
        
        print(f"\n📋 HALO CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred))
        
        # Quick plots
        self.plot_results(cm, y_test, y_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def plot_results(self, cm, y_true, y_proba):
        """Quick plotting for HALO results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1,
                    xticklabels=['No HALO', 'HALO'],
                    yticklabels=['No HALO', 'HALO'])
        ax1.set_title('HALO Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # ROC Curve
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, color='red', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.4f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('HALO ROC Curve')
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
        except:
            ax2.text(0.5, 0.5, 'ROC curve unavailable', ha='center', va='center')
        
        plt.tight_layout()
        
        # Save plots
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig("results/plots/halo_ensemble_results.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results):
        """Save HALO results to file"""
        os.makedirs("results", exist_ok=True)
        filename = "results/halo_ensemble_results.txt"
        
        with open(filename, 'w') as f:
            f.write("HALO Ensemble Model Results\n")
            f.write("="*60 + "\n\n")
            
            f.write("Ensemble Weights:\n")
            for model, weight in self.ensemble_weights.items():
                f.write(f"  {model}: {weight:.3f}\n")
            f.write("\n")
            
            f.write("HALO Performance Metrics:\n")
            f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  F1-Score:  {results['f1_score']:.4f}\n")
            f.write(f"  ROC AUC:   {results['roc_auc']:.4f}\n\n")
            
            f.write("HALO Confusion Matrix:\n")
            f.write(str(results['confusion_matrix']) + "\n\n")
        
        print(f"✅ HALO results saved to {filename}")

def run_halo_ensemble_pipeline(csv_path="merged_data.csv"):
    """Complete fast HALO ensemble pipeline"""
    print("🚀 STARTING FAST HALO ENSEMBLE PIPELINE")
    print("="*60)
    
    # Check if file exists, try different paths
    if not os.path.exists(csv_path):
        alternative_paths = [
            "data/merged_data.csv",
            "merged_data.csv",
            "data\\merged_data.csv"
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                csv_path = alt_path
                print(f"📁 Found data file at: {csv_path}")
                break
        else:
            print(f"❌ Data file not found. Tried: {alternative_paths}")
            return None
    
    total_start_time = time.time()
    
    # Initialize HALO ensemble
    print("🔧 Initializing HALO ensemble...")
    halo_ensemble = HALOEnsemblePredictor()
    
    # Main pipeline progress
    with tqdm(total=6, desc="Pipeline progress", unit="step") as main_pbar:
        # Step 1: Load and preprocess data
        main_pbar.set_description("Loading & preprocessing data")
        df_train, df_test, features = halo_ensemble.load_and_preprocess(csv_path)
        main_pbar.update(1)
        
        # Step 2: Prepare HALO data
        main_pbar.set_description("Preparing HALO data")
        X_train, X_test, y_train, y_test = halo_ensemble.prepare_data(df_train, df_test, features)
        main_pbar.update(1)
        
        # Step 3: Train HALO ensemble
        main_pbar.set_description("Training HALO ensemble")
        halo_ensemble.fit(X_train, y_train)
        main_pbar.update(1)
        
        # Step 4: Evaluate HALO model
        main_pbar.set_description("Evaluating HALO model")
        halo_results = halo_ensemble.evaluate_comprehensive(X_test, y_test)
        main_pbar.update(1)
        
        # Step 5: Save HALO results
        main_pbar.set_description("Saving HALO results")
        halo_ensemble.save_results(halo_results)
        main_pbar.update(1)
        
        # Step 6: Generate final summary
        main_pbar.set_description("Generating final summary")
        total_time = time.time() - total_start_time
        main_pbar.update(1)
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎯 HALO FINAL SUMMARY")
    print("="*60)
    print(f"Total Pipeline Time: {total_time:.2f} seconds")
    print(f"HALO Accuracy:  {halo_results['accuracy']:.4f}")
    print(f"HALO F1-Score:  {halo_results['f1_score']:.4f}")
    print(f"HALO ROC AUC:   {halo_results['roc_auc']:.4f}")
    
    print(f"\n🎉 HALO Pipeline completed successfully!")
    print(f"📁 Check 'results/' directory for detailed HALO outputs")
    
    return halo_results

if __name__ == "__main__":
    # Run the HALO pipeline
    print("🎯 Running HALO-specific Ensemble Pipeline")
    halo_results = run_halo_ensemble_pipeline("merged_data.csv")
