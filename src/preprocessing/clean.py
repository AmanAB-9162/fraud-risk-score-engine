# clean.py
# Basic cleaning: fix types, fill missing, validate ranges.
import pandas as pd
import os

def load_raw(path):
    return pd.read_csv(path)

def clean_df(df):
    df = df.copy()
    # ensure numeric columns
    num_cols = ['time','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    # category
    df['type'] = df['type'].astype(str)
    # clip negative amounts
    df['amount'] = df['amount'].clip(lower=0)
    return df

def save_processed(df, out_dir='data/processed', fname='processed_transactions.csv'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)
    return path
