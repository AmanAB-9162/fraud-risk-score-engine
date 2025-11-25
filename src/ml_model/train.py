# train.py
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from ..preprocessing.feature_engineering import RatioFeatures, get_feature_columns

def load_data(path='data/processed/processed_transactions.csv'):
    return pd.read_csv(path)

def build_pipeline(clf='logistic'):
    numeric_cols, cat_cols = get_feature_columns()
    preproc = ColumnTransformer([('num', StandardScaler(), numeric_cols),
                                 ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])
    if clf == 'logistic':
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
    pipe = Pipeline([('ratio', RatioFeatures()), ('preproc', preproc), ('clf', model)])
    return pipe

def train_and_save(out_dir='models', clf='logistic'):
    os.makedirs(out_dir, exist_ok=True)
    df = load_data()
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline(clf=clf)
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, preds)
    print(f'Validation ROC-AUC: {auc:.4f}')
    joblib.dump(pipe, os.path.join(out_dir, f'{clf}_pipeline.pkl'))
    return os.path.join(out_dir, f'{clf}_pipeline.pkl')

if __name__ == '__main__':
    train_and_save(clf='logistic')
