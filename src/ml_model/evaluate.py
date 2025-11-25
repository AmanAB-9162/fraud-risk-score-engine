# evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import json, os

def evaluate(model_path, data_path='data/processed/processed_transactions.csv', out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    probs = pipe.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds).tolist()
    report = classification_report(y, preds, output_dict=True)
    metrics = {'auc':auc, 'confusion_matrix':cm, 'report':report}
    with open(os.path.join(out_dir,'metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)
    print('AUC:', auc)
    return metrics

if __name__ == '__main__':
    import sys
    evaluate(sys.argv[1] if len(sys.argv)>1 else 'models/logistic_pipeline.pkl')
