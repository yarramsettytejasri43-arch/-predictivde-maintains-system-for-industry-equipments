# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):

    # Create Stress Index (Industry KPI)
    df['stress_index'] = (
        df['temperature'] * 0.3 +
        df['vibration'] * 0.3 +
        df['pressure'] * 0.2 +
        df['load_percentage'] * 0.2
    )

    # Encode plant location
    le = LabelEncoder()
    df['plant_location'] = le.fit_transform(df['plant_location'])

    return df
