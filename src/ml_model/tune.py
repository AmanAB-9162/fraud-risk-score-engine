# tune.py
# Minimal hyperparameter search for RandomForest n_estimators and max_depth
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from .train import build_pipeline, load_data
import itertools

def simple_grid_search(out_dir='models', param_grid=None):
    if param_grid is None:
        param_grid = {'n_estimators':[50,100,200], 'max_depth':[5,10,None]}
    df = load_data()
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best = {'auc':0, 'params':None, 'model_path':None}
    for n, d in itertools.product(param_grid['n_estimators'], param_grid['max_depth']):
        pipe = build_pipeline(clf='rf')
        # set params on RandomForest inside pipeline
        pipe.named_steps['clf'].n_estimators = n
        pipe.named_steps['clf'].max_depth = d
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:,1])
        print(f'n={n} depth={d} -> AUC={auc:.4f}')
        if auc > best['auc']:
            best.update({'auc':auc, 'params':{'n_estimators':n,'max_depth':d}})
            path = os.path.join(out_dir, f'tuned_rf_n{n}_d{d}.pkl')
            os.makedirs(out_dir, exist_ok=True)
            joblib.dump(pipe, path)
            best['model_path'] = path
    print('Best:', best)
    return best

if __name__ == '__main__':
    simple_grid_search()
