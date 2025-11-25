# synthetic_features.py
# Generates synthetic transaction dataset and saves to data/raw/
import os
import numpy as np
import pandas as pd

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def generate(n=5000, seed=42, out_dir='data/raw'):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    time = np.random.randint(0, 60*60*24*30, size=n)
    types = np.random.choice(['PAYMENT','TRANSFER','CASH_OUT','DEBIT','PURCHASE'], size=n, p=[0.5,0.15,0.15,0.1,0.1])
    amount = np.round(np.random.lognormal(mean=3, sigma=1.2, size=n), 2)
    oldbalanceOrg = np.round(np.abs(np.random.normal(loc=2000, scale=3000, size=n)),2)
    oldbalanceDest = np.round(np.abs(np.random.normal(loc=1500, scale=2500, size=n)),2)
    newbalanceOrig = np.maximum(0, oldbalanceOrg - amount + np.random.normal(0,50,size=n))
    newbalanceDest = np.maximum(0, oldbalanceDest + amount + np.random.normal(0,50,size=n))

    type_score = np.array([{'PAYMENT':0,'TRANSFER':1,'CASH_OUT':1.2,'DEBIT':0.2,'PURCHASE':0.1}[t] for t in types])
    rel_amount = amount / (oldbalanceOrg + 1e-6)
    score_raw = -3.0 + 6.0 * np.clip(rel_amount, 0, 10) + 1.5 * type_score + 0.001 * (time % (24*3600))
    score_raw += np.random.normal(0, 1.0, size=n)
    prob = sigmoid(score_raw)
    is_fraud = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        'time': time,
        'type': types,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'is_fraud': is_fraud
    })
    out_path = os.path.join(out_dir, 'synthetic_transactions.csv')
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path} with {n} rows. Fraud rate: {df.is_fraud.mean():.4f}')
    return out_path

if __name__ == '__main__':
    generate()
