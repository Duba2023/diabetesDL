
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from scipy.stats import boxcox, yeojohnson

# --- Load the trained model and scaler ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('diabetes_model.h5')
    return model

@st.cache_resource
def load_scaler():
    scaler = joblib.load('scaler.joblib')
    return scaler

model = load_model()
scaler = load_scaler()

# --- Hardcoded parameters from training notebook for consistency ---
MEDIANS = {
    'Insulin': 125.0,
    'SkinThickness': 29.0,
    'BMI': 32.0,
    'BloodPressure': 72.0,
    'Glucose': 117.0
}

LAMBDAS = {
    'Insulin': 0.0639,
    'DiabetesPedigreeFunction': -0.0731,
    'Age': -1.0944,
    'Pregnancies': 0.1727,
    'BloodPressure': 0.8980
}

# --- Preprocessing function (mimics training notebook steps) ---
def preprocess_input(input_df):
    processed_df = input_df.copy()

    columns_to_process_zeros = ['Insulin', 'SkinThickness', 'BMI', 'BloodPressure', 'Glucose']
    for col in columns_to_process_zeros:
        processed_df[f'{col}_Missing'] = (processed_df[col] == 0).astype(int)
        processed_df[col] = processed_df[col].replace(0, np.nan)
        processed_df[col].fillna(MEDIANS[col], inplace=True)

    columns_for_boxcox = ['Insulin', 'DiabetesPedigreeFunction', 'Age']
    columns_for_yeojohnson = ['Pregnancies', 'BloodPressure']

    for col in columns_for_boxcox:
        processed_df[col] = boxcox(processed_df[col], lmbda=LAMBDAS[col])

    for col in columns_for_yeojohnson:
        processed_df[col] = yeojohnson(processed_df[col], lmbda=LAMBDAS[col])

    columns_to_scale = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    processed_df[columns_to_scale] = scaler.transform(processed_df[columns_to_scale])

    feature_columns_order = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age', 'Insulin_Missing', 'SkinThickness_Missing',
        'BMI_Missing', 'BloodPressure_Missing', 'Glucose_Missing'
    ]
    processed_df = processed_df[feature_columns_order]

    return processed_df

# --- Streamlit UI ---
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("Pima Indians Diabetes Prediction")
st.markdown("""
This app predicts the likelihood of diabetes based on several health indicators.
Please enter the patient's information below:
""")

pregnancies = st.slider("Pregnancies", 0, 17, 3, help="Number of times pregnant")
glucose = st.slider("Glucose (mg/dL)", 0, 199, 120, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 122, 69, help="Diastolic blood pressure (mmHg)")
skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20, help="Triceps skin fold thickness (mm)")
insulin = st.slider("Insulin (mu U/ml)", 0, 846, 79, help="2-Hour serum insulin (mu U/ml)")
bmi = st.slider("BMI (kg/m²)", 0.0, 67.1, 31.9, help="Body mass index (weight in kg/(height in m)^2)")
dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.471, help="Diabetes pedigree function")
age = st.slider("Age (years)", 21, 81, 33, help="Age in years")

input_data = pd.DataFrame([[
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
]], columns=[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
])

if st.button("Predict Diabetes"):
    processed_input = preprocess_input(input_data)
    prediction_proba = model.predict(processed_input)[0][0]

    st.subheader("Prediction Result:")
    if prediction_proba >= 0.5:
        st.error(f"The model predicts **Diabetes** with a probability of {prediction_proba:.2f}")
    else:
        st.success(f"The model predicts **No Diabetes** with a probability of {prediction_proba:.2f}")

    st.markdown("""
    **Disclaimer:** This is a predictive model based on a specific dataset and should not be used as a substitute for professional medical advice.
    """)
