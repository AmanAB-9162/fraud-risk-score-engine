# feature_engineering.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RatioFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['rel_amount_org'] = X['amount'] / (X['oldbalanceOrg'] + 1e-6)
        X['delta_org'] = X['oldbalanceOrg'] - X['newbalanceOrig']
        X['delta_dest'] = X['newbalanceDest'] - X['oldbalanceDest']
        X['hour'] = (X['time'] // 3600) % 24
        return X

def get_feature_columns():
    numeric = ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','rel_amount_org','delta_org','delta_dest','hour']
    categorical = ['type']
    return numeric, categorical
