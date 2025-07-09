# AutoGluon ML Pipeline for Google Colab
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, classification_report
from autogluon.tabular import TabularPredictor

def load_and_preprocess(csv_path):
    """Load and preprocess the data"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Initial shape: {df.shape}")
    
    if 'epoch_for_cdf_mod' in df.columns:
        df['epoch_for_cdf_mod'] = pd.to_datetime(df['epoch_for_cdf_mod'])
    
    df.replace(-1e31, np.nan, inplace=True)
    df = df.sort_values('epoch_for_cdf_mod')
    df.dropna(thresh=len(df.columns) * 0.7, inplace=True)
    df = df.ffill().bfill()
    print(f"After cleaning: {df.shape}")
    
    if 'epoch_for_cdf_mod' in df.columns:
        df['hour'] = df['epoch_for_cdf_mod'].dt.hour
        df['weekday'] = df['epoch_for_cdf_mod'].dt.weekday
    
    columns_to_remove = ['epoch_for_cdf_mod', 'spacecraft_xpos', 'spacecraft_ypos', 'spacecraft_zpos']
    features = [col for col in df.columns if col not in columns_to_remove + ['CME', 'HALO']]
    print(f"Feature columns: {features}")
    
    return df, features

def run_autogluon(csv_path):
    df, features = load_and_preprocess(csv_path)
    
    # Split data
    split_time = df['epoch_for_cdf_mod'].quantile(0.8)
    train = df[df['epoch_for_cdf_mod'] < split_time]
    test = df[df['epoch_for_cdf_mod'] >= split_time]
    
    print("Training CME model...")
    predictor_cme = TabularPredictor(label='CME').fit(train[features + ['CME']])
    pred_cme = predictor_cme.predict(test[features])
    print("AutoGluon CME F1:", f1_score(test['CME'], pred_cme, average='weighted'))
    
    print("Training HALO model...")
    predictor_halo = TabularPredictor(label='HALO').fit(train[features + ['HALO']])
    pred_halo = predictor_halo.predict(test[features])
    print("AutoGluon HALO F1:", f1_score(test['HALO'], pred_halo, average='weighted'))

if __name__ == "__main__":
    run_autogluon("merged_data.csv")
