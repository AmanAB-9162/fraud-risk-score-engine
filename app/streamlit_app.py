# streamlit_app.py
import streamlit as st
import pandas as pd, joblib, os
st.title('Fraud Risk Scoring Engine - Demo')
st.write('Upload transactions CSV (same columns as processed) to score.')
uploaded = st.file_uploader('CSV', type=['csv'])
model_path = st.text_input('Model path', value='models/logistic_pipeline.pkl')
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if st.button('Score'):
        if os.path.exists(model_path):
            pipe = joblib.load(model_path)
            probs = pipe.predict_proba(df)[:,1]
            df['fraud_score'] = probs
            st.dataframe(df.head(50))
        else:
            st.error('Model not found at ' + model_path)
