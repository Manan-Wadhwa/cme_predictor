#%% Preprocessing Utilities
import pandas as pd
import numpy as np

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.replace(-1e31, np.nan, inplace=True)
    df['epoch_for_cdf_mod'] = pd.to_datetime(df['epoch_for_cdf_mod'])

    # Time features
    df['hour'] = df['epoch_for_cdf_mod'].dt.hour
    df['minute'] = df['epoch_for_cdf_mod'].dt.minute
    df['weekday'] = df['epoch_for_cdf_mod'].dt.weekday

    # Fill NaNs
    df.fillna(df.median(), inplace=True)

    # Features and targets
    features = [col for col in df.columns if col not in ['epoch_for_cdf_mod', 'CME', 'HALO']]
    return df, features
