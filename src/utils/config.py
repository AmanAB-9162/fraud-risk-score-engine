# config.py
import os
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../'))
RAW = os.path.join(BASE, 'data/raw/synthetic_transactions.csv')
PROCESSED = os.path.join(BASE, 'data/processed/processed_transactions.csv')
MODELS = os.path.join(BASE, 'models')
RESULTS = os.path.join(BASE, 'results')
