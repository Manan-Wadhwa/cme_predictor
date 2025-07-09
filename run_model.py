#!/usr/bin/env python3
"""
Simple script to load and run predictions with AutoGluon pickle model
"""

import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_pickle_model(model_path):
    """Load a pickle model file"""
    try:
        print(f"🔄 Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_data(data_path):
    """Load data for prediction"""
    try:
        print(f"🔄 Loading data from: {data_path}")
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.pkl'):
            data = pd.read_pickle(data_path)
        else:
            print(f"❌ Unsupported file format: {data_path}")
            return None
        
        print(f"✅ Data loaded: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def run_predictions(model, data):
    """Run predictions on the data"""
    try:
        print("🔮 Running predictions...")
        
        # Try different prediction methods
        if hasattr(model, 'predict'):
            predictions = model.predict(data)
        elif hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(data)
        else:
            print("❌ Model doesn't have predict or predict_proba method")
            return None
        
        print(f"✅ Predictions completed! Shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
        return predictions
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

def main():
    """Main function to run the pickle model"""
    print("🚀 AutoGluon Pickle Model Runner")
    print("=" * 50)
    
    # Check common locations for model and data files
    possible_model_paths = [
        "model.pkl",
        "autogluon_model.pkl", 
        "best_model.pkl",
        "downloads/model.pkl",
        "downloads/autogluon_model.pkl",
        os.path.expanduser("~/Downloads/model.pkl"),
        os.path.expanduser("~/Downloads/autogluon_model.pkl")
    ]
    
    possible_data_paths = [
        "merged_data.csv",
        "test_data.csv",
        "data.csv",
        "downloads/merged_data.csv",
        "downloads/test_data.csv",
        os.path.expanduser("~/Downloads/merged_data.csv"),
        os.path.expanduser("~/Downloads/test_data.csv")
    ]
    
    # Find model file
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No model file found. Please specify the path:")
        model_path = input("Enter model file path: ").strip()
        if not os.path.exists(model_path):
            print(f"❌ File not found: {model_path}")
            return
    
    # Find data file
    data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print("❌ No data file found. Please specify the path:")
        data_path = input("Enter data file path: ").strip()
        if not os.path.exists(data_path):
            print(f"❌ File not found: {data_path}")
            return
    
    # Load model
    model = load_pickle_model(model_path)
    if model is None:
        return
    
    # Load data
    data = load_data(data_path)
    if data is None:
        return
    
    # Display model info
    print(f"\n📋 Model Information:")
    print(f"   Type: {type(model)}")
    if hasattr(model, '__dict__'):
        print(f"   Attributes: {list(model.__dict__.keys())[:5]}...")  # Show first 5 attributes
    
    # Display data info
    print(f"\n📊 Data Information:")
    print(f"   Shape: {data.shape}")
    print(f"   Columns: {list(data.columns)[:10]}...")  # Show first 10 columns
    
    # Run predictions
    predictions = run_predictions(model, data)
    if predictions is None:
        return
    
    # Save predictions
    output_path = "predictions.csv"
    try:
        if isinstance(predictions, np.ndarray):
            pred_df = pd.DataFrame(predictions, columns=[f'prediction_{i}' for i in range(predictions.shape[1])] if len(predictions.shape) > 1 else ['prediction'])
        else:
            pred_df = pd.DataFrame({'prediction': predictions})
        
        pred_df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to: {output_path}")
        
        # Show first few predictions
        print(f"\n🔍 First 5 predictions:")
        print(pred_df.head())
        
    except Exception as e:
        print(f"⚠️  Could not save predictions: {e}")
        print(f"Predictions: {predictions[:5] if hasattr(predictions, '__getitem__') else predictions}")
    
    print(f"\n🎉 Complete!")

if __name__ == "__main__":
    main()
