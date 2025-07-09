# Complete Ensemble Model with Data Processing and Comprehensive Evaluation
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    def __init__(self):
        # Define ensemble weights as specified
        self.ensemble_weights = {
            'RandomForestEntr': 0.333,
            'ExtraTreesEntr': 0.333,
            'NeuralNetTorch': 0.167,
            'LightGBMLarge': 0.167
        }
        
        # Initialize models
        self.models = {
            'RandomForestEntr': RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTreesEntr': ExtraTreesClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'NeuralNetTorch': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            'LightGBMLarge': lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=15,
                num_leaves=100,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbose=-1
            )
        }
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def load_and_preprocess(self, csv_path):
        """Load and preprocess the data"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Initial shape: {df.shape}")
        
        # Handle datetime
        if 'epoch_for_cdf_mod' in df.columns:
            print("Parsing 'epoch_for_cdf_mod' as datetime...")
            df['epoch_for_cdf_mod'] = pd.to_datetime(df['epoch_for_cdf_mod'])
        
        # Clean data
        df.replace(-1e31, np.nan, inplace=True)
        df = df.sort_values('epoch_for_cdf_mod')
        df.dropna(thresh=len(df.columns) * 0.7, inplace=True)
        df = df.ffill().bfill()
        print(f"After cleaning: {df.shape}")
        
        # Feature engineering
        if 'epoch_for_cdf_mod' in df.columns:
            df['hour'] = df['epoch_for_cdf_mod'].dt.hour
            df['weekday'] = df['epoch_for_cdf_mod'].dt.weekday
            df['month'] = df['epoch_for_cdf_mod'].dt.month
            df['day_of_year'] = df['epoch_for_cdf_mod'].dt.dayofyear
        
        # Define features
        columns_to_remove = ['epoch_for_cdf_mod', 'spacecraft_xpos', 'spacecraft_ypos', 'spacecraft_zpos']
        target_columns = ['CME', 'HALO']
        
        features = [col for col in df.columns if col not in columns_to_remove + target_columns]
        print(f"Feature columns ({len(features)}): {features}")
        
        return df, features
    
    def prepare_data(self, df, features, target_col='CME', test_size=0.2):
        """Prepare data for training"""
        print(f"\nPreparing data for target: {target_col}")
        
        # Split data temporally
        split_time = df['epoch_for_cdf_mod'].quantile(0.8)
        train_df = df[df['epoch_for_cdf_mod'] < split_time]
        test_df = df[df['epoch_for_cdf_mod'] >= split_time]
        
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Class distribution in train: {y_train.value_counts().to_dict()}")
        print(f"Class distribution in test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train, y_train):
        """Train all models in the ensemble"""
        print("\n" + "="*50)
        print("TRAINING ENSEMBLE MODELS")
        print("="*50)
        
        # Scale features for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            if model_name == 'NeuralNetTorch':
                # Use scaled features for neural network
                model.fit(X_train_scaled, y_train)
            else:
                # Use original features for tree-based models
                model.fit(X_train, y_train)
            
            print(f"✅ {model_name} training completed!")
        
        self.is_fitted = True
        print("\n🎉 All models trained successfully!")
    
    def predict_proba(self, X):
        """Get ensemble predictions with probabilities"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction!")
        
        # Scale features for neural network
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        ensemble_proba = np.zeros((len(X), 2))  # Assuming binary classification
        
        for model_name, model in self.models.items():
            weight = self.ensemble_weights[model_name]
            
            if model_name == 'NeuralNetTorch':
                proba = model.predict_proba(X_scaled)
            else:
                proba = model.predict_proba(X)
            
            ensemble_proba += weight * proba
            print(f"{model_name} (weight: {weight:.3f}) - predictions obtained")
        
        return ensemble_proba
    
    def predict(self, X):
        """Get ensemble predictions"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def evaluate_comprehensive(self, X_test, y_test, target_name="Target"):
        """Comprehensive evaluation with all metrics"""
        print(f"\n" + "="*60)
        print(f"COMPREHENSIVE EVALUATION - {target_name}")
        print("="*60)
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = 0.0
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"📊 PERFORMANCE METRICS:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        
        print(f"\n📈 CONFUSION MATRIX:")
        print(cm)
        
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, target_name)
        
        # Plot ROC curve
        self.plot_roc_curve(y_test, y_proba, target_name)
        
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
    
    def plot_confusion_matrix(self, cm, target_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Event', 'Event'],
                    yticklabels=['No Event', 'Event'])
        plt.title(f'Confusion Matrix - {target_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        os.makedirs("results/plots", exist_ok=True)
        plt.savefig(f"results/plots/confusion_matrix_{target_name.lower()}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_proba, target_name):
        """Plot ROC curve"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {target_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f"results/plots/roc_curve_{target_name.lower()}.png", dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not plot ROC curve: {e}")
    
    def save_results(self, results, target_name, filename=None):
        """Save results to file"""
        if filename is None:
            filename = f"results/ensemble_{target_name.lower()}_results.txt"
        
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write(f"Ensemble Model Results - {target_name}\n")
            f.write("="*60 + "\n\n")
            
            f.write("Ensemble Weights:\n")
            for model, weight in self.ensemble_weights.items():
                f.write(f"  {model}: {weight:.3f}\n")
            f.write("\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  F1-Score:  {results['f1_score']:.4f}\n")
            f.write(f"  ROC AUC:   {results['roc_auc']:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(str(results['confusion_matrix']) + "\n\n")
        
        print(f"Results saved to {filename}")

def run_ensemble_pipeline(csv_path="merged_data.csv"):
    """Complete ensemble pipeline"""
    print("🚀 STARTING ENSEMBLE PREDICTION PIPELINE")
    print("="*60)
    
    # Initialize ensemble
    ensemble = EnsemblePredictor()
    
    # Load and preprocess data
    df, features = ensemble.load_and_preprocess(csv_path)
    
    results = {}
    
    # Train and evaluate for both targets
    for target in ['CME', 'HALO']:
        print(f"\n{'='*60}")
        print(f"PROCESSING TARGET: {target}")
        print("="*60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = ensemble.prepare_data(df, features, target)
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        target_results = ensemble.evaluate_comprehensive(X_test, y_test, target)
        
        # Save results
        ensemble.save_results(target_results, target)
        
        results[target] = target_results
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    
    for target, target_results in results.items():
        print(f"\n{target} Results:")
        print(f"  Accuracy: {target_results['accuracy']:.4f}")
        print(f"  F1-Score: {target_results['f1_score']:.4f}")
        print(f"  ROC AUC:  {target_results['roc_auc']:.4f}")
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📁 Check 'results/' directory for detailed outputs and plots")
    
    return results

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_ensemble_pipeline("merged_data.csv")
