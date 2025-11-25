# shap_explain.py
import shap
import joblib
import pandas as pd
import os

def global_shap(model_path, data_path='data/processed/processed_transactions.csv', out_dir='results/shap_plots'):
    os.makedirs(out_dir, exist_ok=True)
    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop(columns=['is_fraud'])
    # get transformed data for SHAP: use pipeline up to preproc if available
    # shap can accept a function; we will pass a wrapper that takes raw X and outputs probabilities
    explainer = shap.Explainer(lambda x: pipe.predict_proba(x)[:,1], X, feature_names=X.columns.tolist())
    shap_values = explainer(X)
    # save summary plot
    import matplotlib.pyplot as plt
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'shap_summary.png'))
    print('Saved SHAP summary to', out_dir)
    return os.path.join(out_dir,'shap_summary.png')
