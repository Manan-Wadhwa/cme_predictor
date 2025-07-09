import pandas as pd
import numpy as np

def load_and_preprocess(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Initial shape: {df.shape}")
    if 'epoch_for_cdf_mod' in df.columns:
        print("Parsing 'epoch_for_cdf_mod' as datetime...")
        df['epoch_for_cdf_mod'] = pd.to_datetime(df['epoch_for_cdf_mod'])
    df.replace(-1e31, np.nan, inplace=True)
    df = df.sort_values('epoch_for_cdf_mod')
    df.dropna(thresh=len(df.columns) * 0.7, inplace=True)
    df = df.ffill().bfill()
    print(f"After cleaning: {df.shape}")
    # Feature engineering
    if 'epoch_for_cdf_mod' in df.columns:
        df['hour'] = df['epoch_for_cdf_mod'].dt.hour
        df['weekday'] = df['epoch_for_cdf_mod'].dt.weekday
    columns_to_remove = ['epoch_for_cdf_mod', 'spacecraft_xpos', 'spacecraft_ypos', 'spacecraft_zpos']
    features = [col for col in df.columns if col not in columns_to_remove + ['CME', 'HALO']]
    print(f"Feature columns: {features}")
    return df, features
