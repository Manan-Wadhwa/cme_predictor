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
        
        # Initialize optimized models for speed
        self.models = {
            'RandomForestEntr': RandomForestClassifier(
                n_estimators=300,  # Reduced for speed
                max_depth=15,      # Reduced for speed
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                warm_start=True
            ),
            'ExtraTreesEntr': ExtraTreesClassifier(
                n_estimators=300,  # Reduced for speed
                max_depth=15,      # Reduced for speed
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                warm_start=True
            ),
            'NeuralNetTorch': MLPClassifier(
                hidden_layer_sizes=(128, 64),  # Reduced for speed
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=300,  # Reduced for speed
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'LightGBMLarge': lgb.LGBMClassifier(
                n_estimators=500,   # Reduced for speed
                learning_rate=0.1,  # Increased for speed
                max_depth=10,       # Reduced for speed
                num_leaves=50,      # Reduced for speed
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        }
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_and_preprocess(self, csv_path):
        """Fast data loading and preprocessing for HALO"""
        print(f"🔄 Loading HALO data from {csv_path}...")
        
        # Use efficient data types
        dtype_dict = {
            'CME': 'int8',
            'HALO': 'int8'
        }
        
        df = pd.read_csv(csv_path, dtype=dtype_dict)
        print(f"Initial shape: {df.shape}")
        
        # Handle datetime efficiently
        if 'epoch_for_cdf_mod' in df.columns:
            df['epoch_for_cdf_mod'] = pd.to_datetime(df['epoch_for_cdf_mod'], cache=True)
        
        # Fast data cleaning
        df.replace(-1e31, np.nan, inplace=True)
        df = df.sort_values('epoch_for_cdf_mod')
        
        # More aggressive cleaning for speed
        df.dropna(thresh=len(df.columns) * 0.5, inplace=True)
        df = df.fillna(method='ffill').fillna(method='bfill')
        print(f"After cleaning: {df.shape}")
        
        # Essential feature engineering only
        if 'epoch_for_cdf_mod' in df.columns:
            df['hour'] = df['epoch_for_cdf_mod'].dt.hour
            df['weekday'] = df['epoch_for_cdf_mod'].dt.weekday
        
        # Define features (excluding CME since we're predicting HALO)
        columns_to_remove = ['epoch_for_cdf_mod', 'spacecraft_xpos', 'spacecraft_ypos', 'spacecraft_zpos', 'CME']
        features = [col for col in df.columns if col not in columns_to_remove + ['HALO']]
        print(f"HALO Feature columns ({len(features)}): {features}")
        
        return df, features
    
    def prepare_data(self, df, features):
        """Prepare HALO data for training"""
        print(f"🎯 Preparing HALO data for training...")
        
        # Temporal split
        split_time = df['epoch_for_cdf_mod'].quantile(0.8)
        train_df = df[df['epoch_for_cdf_mod'] < split_time]
        test_df = df[df['epoch_for_cdf_mod'] >= split_time]
        
        X_train = train_df[features].values  # Convert to numpy for speed
        y_train = train_df['HALO'].values
        X_test = test_df[features].values
        y_test = test_df['HALO'].values
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"HALO distribution in train: {np.bincount(y_train)}")
        print(f"HALO distribution in test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train, y_train):
        """Fast training of HALO ensemble models"""
        print("\n" + "="*50)
        print("🚀 FAST TRAINING HALO ENSEMBLE MODELS")
        print("="*50)
        
        # Scale features once
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train models with progress tracking
        for i, (model_name, model) in enumerate(self.models.items(), 1):
            print(f"[{i}/4] Training {model_name} for HALO...")
            
            if model_name == 'NeuralNetTorch':
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            
            print(f"✅ {model_name} completed!")
        
        self.is_fitted = True
        print("\n🎉 HALO ensemble training completed!")
    
    def predict_proba(self, X):
        """Fast ensemble predictions with probabilities"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction!")
        
        X_scaled = self.scaler.transform(X)
        ensemble_proba = np.zeros((len(X), 2))
        
        for model_name, model in self.models.items():
            weight = self.ensemble_weights[model_name]
            
            if model_name == 'NeuralNetTorch':
                proba = model.predict_proba(X_scaled)
            else:
                proba = model.predict_proba(X)
            
            ensemble_proba += weight * proba
        
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
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = 0.0
        
        cm = confusion_matrix(y_test, y_pred)
        
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

def run_halo_ensemble_pipeline(csv_path="data/merged_data.csv"):
    """Complete fast HALO ensemble pipeline"""
    print("🚀 STARTING FAST HALO ENSEMBLE PIPELINE")
    print("="*60)
    
    # Initialize HALO ensemble
    halo_ensemble = HALOEnsemblePredictor()
    
    # Load and preprocess data
    df, features = halo_ensemble.load_and_preprocess(csv_path)
    
    # Prepare HALO data
    X_train, X_test, y_train, y_test = halo_ensemble.prepare_data(df, features)
    
    # Train HALO ensemble
    halo_ensemble.fit(X_train, y_train)
    
    # Evaluate HALO model
    halo_results = halo_ensemble.evaluate_comprehensive(X_test, y_test)
    
    # Save HALO results
    halo_ensemble.save_results(halo_results)
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎯 HALO FINAL SUMMARY")
    print("="*60)
    print(f"HALO Accuracy:  {halo_results['accuracy']:.4f}")
    print(f"HALO F1-Score:  {halo_results['f1_score']:.4f}")
    print(f"HALO ROC AUC:   {halo_results['roc_auc']:.4f}")
    
    print(f"\n🎉 HALO Pipeline completed successfully!")
    print(f"📁 Check 'results/' directory for detailed HALO outputs")
    
    return halo_results

if __name__ == "__main__":
    # Run the HALO pipeline
    print("🎯 Running HALO-specific Ensemble Pipeline")
    halo_results = run_halo_ensemble_pipeline("data/merged_data.csv")
