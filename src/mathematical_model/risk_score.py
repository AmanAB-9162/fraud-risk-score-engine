# risk_score.py
# A simple interpretable mathematical risk score (log-odds style)
import numpy as np

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def compute_risk_score_row(row, weights=None, intercept=-3.0):
    # row: dict-like with amount, oldbalanceOrg, type, time
    if weights is None:
        weights = {
            'rel_amount_org': 6.0,
            'type_transfer': 1.0,
            'type_cash_out': 1.2,
            'time_hour': 0.001
        }
    rel = row.get('amount',0) / (row.get('oldbalanceOrg',1e-6))
    type_map = {'TRANSFER':weights.get('type_transfer',1.0),
                'CASH_OUT':weights.get('type_cash_out',1.2)}
    tscore = type_map.get(row.get('type','PAYMENT'), 0.0)
    hour = (int(row.get('time',0)) // 3600) % 24
    raw = intercept + weights.get('rel_amount_org',6.0) * np.clip(rel,0,10) + tscore + weights.get('time_hour',0.001)*(hour)
    prob = sigmoid(raw)
    return {'raw_score': raw, 'probability': float(prob)}

def compute_risk_scores(df, weights=None, intercept=-3.0):
    out = df.copy()
    scores = out.apply(lambda r: compute_risk_score_row(r, weights=weights, intercept=intercept), axis=1)
    out['math_raw'] = [s['raw_score'] for s in scores]
    out['math_prob'] = [s['probability'] for s in scores]
    return out
