import json
import os
import joblib
import pandas as pd
import streamlit as st

# ---------- Load schema ----------
def load_schema():
    path = "artifacts/schema.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"num_cols": [], "all_cols": []}

schema = load_schema()
num_cols = schema.get("num_cols", [])
all_cols = schema.get("all_cols", [])

# ---------- Load artifacts ----------
preproc_path = "artifacts/preprocessor.pkl"
encoder_path = "artifacts/encoder.pkl"
model_path = "artifacts/model.pkl"
feature_cols_path = "artifacts/feature_columns.json"

missing_files = [p for p in [preproc_path, encoder_path, model_path] if not os.path.exists(p)]
if missing_files:
    st.error(f"Missing artifact(s): {missing_files}")

scaler = joblib.load(preproc_path)
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)

# Load feature columns if exists
if os.path.exists(feature_cols_path):
    with open(feature_cols_path, "r") as f:
        final_features = json.load(f)
else:
    final_features = None

# ---------- Streamlit UI ----------
st.title("Fraud Detection")

with st.form("fraud_form"):
    inputs = {}
    for col in all_cols:
        if col == "hour_of_day":
            inputs[col] = st.selectbox(col, options=list(range(24)), index=0)
        elif col == "day_of_week":
            inputs[col] = st.selectbox(col, options=list(range(7)), index=0)
        else:
            inputs[col] = st.number_input(col, value=0.0, format="%f")
    submitted = st.form_submit_button("Predict")

# ---------- Prediction ----------
if submitted:
    df = pd.DataFrame([inputs])

    # Scale numeric columns
    if len(num_cols) > 0:
        df[num_cols] = scaler.transform(df[num_cols])

    # Encode categorical columns
    categorical_cols = ['hour_of_day', 'day_of_week']
    df_cat = encoder.transform(df[categorical_cols])
    df_cat = pd.DataFrame(df_cat, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine numeric and categorical
    df_final = pd.concat([df[num_cols], df_cat], axis=1)

    # Align columns to match training
    if final_features is not None:
        df_final = df_final.reindex(columns=final_features, fill_value=0)

    # Make prediction
    prob = model.predict_proba(df_final)[0][1]
    label = "FRAUD" if prob >= 0.2 else "NOT FRAUD"

    # Show results
    st.write("Probability:", float(prob))
    st.write("Prediction:", label)
