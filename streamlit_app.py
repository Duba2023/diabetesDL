
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from scipy.stats import boxcox, yeojohnson

# --- Configuration --- #
MODEL_PATH = 'keras_model.h5'
SCALER_PATH = 'scaler.joblib'
CALIBRATOR_PATH = 'calibrator.joblib'

# Feature definitions (must match training order and names)
FEATURE_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age', 'Insulin_Missing', 'SkinThickness_Missing',
    'BMI_Missing', 'BloodPressure_Missing', 'Glucose_Missing'
]

COLUMNS_TO_PROCESS_ZERO_TO_NAN = ['Insulin', 'SkinThickness', 'BMI', 'BloodPressure', 'Glucose']

# Medians for imputation (calculated from training data, after 0->NaN replacement)
MEDIANS_FOR_IMPUTATION = {
    'Glucose': 117.0,
    'BloodPressure': 72.0,
    'SkinThickness': 29.0,
    'Insulin': 125.0,
    'BMI': 32.3
}

COLUMNS_FOR_BOXCOX = ['Insulin', 'DiabetesPedigreeFunction', 'Age']
LAMBDAS_BOXCOX = {
    'Insulin': 0.0639,
    'DiabetesPedigreeFunction': -0.0731,
    'Age': -1.0944
}

COLUMNS_FOR_YEOJOHNSON = ['Pregnancies', 'BloodPressure']
LAMBDAS_YEOJOHNSON = {
    'Pregnancies': 0.1727,
    'BloodPressure': 0.8980
}

COLUMNS_TO_SCALE = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# --- Load Assets --- #
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_scaler():
    scaler = joblib.load(SCALER_PATH)
    return scaler

@st.cache_resource
def load_calibrator():
    calibrator = joblib.load(CALIBRATOR_PATH)
    return calibrator

model = load_model()
scaler = load_scaler()
calibrator = load_calibrator()

# --- Preprocessing Function --- #
def preprocess_input(input_df):
    processed_df = input_df.copy()

    # 1. Handle 0 to NaN, create missing indicators, and impute with median
    for col in COLUMNS_TO_PROCESS_ZERO_TO_NAN:
        processed_df[f'{col}_Missing'] = (processed_df[col] == 0).astype(int)
        processed_df[col] = processed_df[col].replace(0, np.nan)
        if processed_df[col].isnull().any(): # Should always be true if 0s were replaced
            median_val = MEDIANS_FOR_IMPUTATION[col]
            processed_df[col].fillna(median_val, inplace=True)

    # 2. Apply skewness transformations
    for col in COLUMNS_FOR_BOXCOX:
        processed_df[col] = boxcox(processed_df[col], lmbda=LAMBDAS_BOXCOX[col])

    for col in COLUMNS_FOR_YEOJOHNSON:
        processed_df[col] = yeojohnson(processed_df[col], lmbda=LAMBDAS_YEOJOHNSON[col])

    # Ensure all feature columns exist, add missing indicator columns if they don't already
    for col in FEATURE_COLUMNS:
        if col not in processed_df.columns:
            processed_df[col] = 0 # Default to 0 for missing indicators if no 0s were present

    # Reorder columns to match training data
    processed_df = processed_df[FEATURE_COLUMNS]

    # 3. Scale numerical features
    processed_df[COLUMNS_TO_SCALE] = scaler.transform(processed_df[COLUMNS_TO_SCALE])

    return processed_df

# --- Streamlit App --- #
st.set_page_config(page_title='Diabetes Prediction App', layout='centered')
st.title('Pima Indians Diabetes Prediction')
st.markdown('Enter patient details to predict the likelihood of diabetes.')

# Input fields
with st.form('prediction_form'):
    st.header('Patient Information')
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=17, value=1)
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=199, value=120)
    blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=122, value=70)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=99, value=20)
    insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=846, value=79)
    bmi = st.number_input('BMI (kg/m^2)', min_value=0.0, max_value=67.1, value=30.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.471, format="%.3f")
    age = st.number_input('Age (years)', min_value=21, max_value=81, value=30)

    submit_button = st.form_submit_button('Predict Diabetes')

if submit_button:
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }])

    # Preprocess the input data
    processed_input = preprocess_input(input_data)

    # Make prediction with the Keras model
    raw_prediction_proba = model.predict(processed_input)[0][0]

    # Calibrate the prediction
    calibrated_prediction_proba = calibrator.predict_proba(np.array(raw_prediction_proba).reshape(-1, 1))[0][1]

    # Define the optimal threshold (from training)
    optimal_threshold = 0.364 # This value comes from the previous training output: best_threshold
    prediction = (calibrated_prediction_proba >= optimal_threshold).astype(int)

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(f'The patient is likely to have diabetes with a probability of {calibrated_prediction_proba:.2f}.')
    else:
        st.success(f'The patient is unlikely to have diabetes with a probability of {calibrated_prediction_proba:.2f}.')

    st.markdown(f"*Raw Model Probability: {raw_prediction_proba:.2f}*")
    st.markdown(f"*Calibrated Probability: {calibrated_prediction_proba:.2f}*")
    st.markdown(f"*Decision Threshold: {optimal_threshold:.3f}*")

st.subheader('Instructions to run the app:')
st.code('1. Save the code above as `streamlit_app.py`
2. Make sure `keras_model.h5`, `scaler.joblib`, and `calibrator.joblib` are in the same directory.
3. Run `pip install streamlit pandas numpy tensorflow scikit-learn joblib scipy`
4. Open your terminal, navigate to the directory containing the file, and run `streamlit run streamlit_app.py`')
