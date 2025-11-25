
# streamlit_app.py
import streamlit as st
import sys, os

# -------------------------------
# FIX 1: Add project root to PYTHON PATH
# -------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # app/
PROJECT_DIR = os.path.dirname(ROOT_DIR)               # fraud-risk-score-engine/
sys.path.append(PROJECT_DIR)

# Now imports will work
from src.ml_model.train import train_and_save
import pandas as pd
import joblib

st.title("Fraud Risk Scoring Engine - Demo")

# -------------------------------
# TRAIN MODEL BUTTON
# -------------------------------
if st.button("Train Model in Cloud"):
    path = train_and_save()
    st.success(f"Model trained and saved at: {path}")

# -------------------------------
# SCORE CSV
# -------------------------------
st.write("Upload transactions CSV to score")
uploaded = st.file_uploader("CSV", type=["csv"])

model_path = "models/logistic_pipeline.pkl"

if uploaded:
    df = pd.read_csv(uploaded)

    if st.button("Score"):
        if os.path.exists(model_path):
            try:
                pipe = joblib.load(model_path)
                df["fraud_score"] = pipe.predict_proba(df)[:, 1]
                st.dataframe(df.head(50))

            except Exception as e:
                st.error("Error loading model: " + str(e))
        else:
            st.warning("Model not found. Please click 'Train Model in Cloud' first.")

# # # streamlit_app.py
# # import streamlit as st

# # if st.button("Train Model in Cloud"):
# #     from src.ml_model.train import train_and_save
# #     path = train_and_save()
# #     st.success("Model trained and saved: " + path)

# # import pandas as pd, joblib, os
# # st.title('Fraud Risk Scoring Engine - Demo')
# # st.write('Upload transactions CSV (same columns as processed) to score.')
# # uploaded = st.file_uploader('CSV', type=['csv'])
# # model_path = st.text_input('Model path', value='models/logistic_pipeline.pkl')
# # if uploaded is not None:
# #     df = pd.read_csv(uploaded)
# #     if st.button('Score'):
# #         if os.path.exists(model_path):
# #             pipe = joblib.load(model_path)
# #             probs = pipe.predict_proba(df)[:,1]
# #             df['fraud_score'] = probs
# #             st.dataframe(df.head(50))
# #         else:
# #             st.error('Model not found at ' + model_path)

# import streamlit as st
# import pandas as pd
# import os

# st.title("Fraud Risk Scoring Engine - Cloud Demo")

# # ----------------------------
# # 1) TRAIN MODEL INSIDE STREAMLIT CLOUD
# # ----------------------------
# if st.button("📌 Train Model (Cloud)"):
#     st.write("Training model... Please wait ⏳")

#     # local import (IMPORTANT)
#     from src.ml_model.train import train_and_save
    
#     model_path = train_and_save()
#     st.success(f"✅ Model trained and saved at: {model_path}")


# # ----------------------------
# # 2) SCORE UPLOADED CSV
# # ----------------------------
# st.header("Upload CSV to Score Fraud Risk")

# uploaded = st.file_uploader("Upload processed CSV", type=["csv"])

# model_path = "models/logistic_pipeline.pkl"   # FIXED PATH


# if uploaded is not None:
#     df = pd.read_csv(uploaded)
#     st.write("Preview:", df.head())

#     if st.button("🚀 Score CSV"):
#         if not os.path.exists(model_path):
#             st.error("❌ Model not found. Please train the model first.")
#         else:
#             import joblib
#             pipe = joblib.load(model_path)

#             probs = pipe.predict_proba(df)[:, 1]
#             df["fraud_score"] = probs
#             st.success("✅ Scoring completed!")
#             st.dataframe(df)
