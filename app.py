import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Load the pre-trained model and scaler ---
rf_model = joblib.load('random_forest_diabetes.pkl')
scaler = joblib.load('scaler.pkl')

# --- Hardcoded median values for imputation ---
median_values = {
    'Glucose': 117.0,
    'BloodPressure': 72.0,
    'SkinThickness': 29.0,
    'Insulin': 125.0, 
    'BMI': 32.4
}
# --- Preprocessing Function ---
def preprocess_input(df_row):
    df_processed = df_row.copy()
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace 0 with NaN
    for col in cols_to_impute:
        if df_processed[col] == 0:
            df_processed[col] = np.nan

    # Fill missing values with hardcoded medians
    for col in cols_to_impute:
        if pd.isna(df_processed[col]):
            df_processed[col] = median_values[col]

    # Log transform 'Insulin' and 'DiabetesPedigreeFunction'
    df_processed['Insulin'] = np.log1p(df_processed['Insulin'])
    df_processed['DiabetesPedigreeFunction'] = np.log1p(df_processed['DiabetesPedigreeFunction'])

    return df_processed

# --- Feature Engineering Functions ---
def bmi_category(BMI):
    if BMI < 18.5:
        return 0   # Underweight
    elif BMI < 25:
        return 1   # Normal
    elif BMI < 30:
        return 2   # Overweight
    else:
        return 3   # Obese

def glucose_risk(Glucose):
    if Glucose < 100:
        return 0
    elif Glucose < 126:
        return 1
    else:
        return 2

def age_group(Age):
    if Age < 30:
        return 0
    elif Age < 45:
        return 1
    else:
        return 2

# --- Streamlit App Setup ---
st.markdown(
    """
    <h1 style='text-align: center; color: #1f77b4; font-size: 50px; font-weight: bold;'>
        🩺 Diabetes Prediction App
    </h1>
    <hr style="border:2px solid #1f77b4;">
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-size: 17px;
        line-height: 1.7;
    }

    .stMarkdown, .stText, .stCaption, p, label, input, textarea {
        line-height: 1.7;
    }

   .main-header {
    font-size: 40px !important; 
    color: #3776AB;
    text-align: center;
    margin-bottom: 25px;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    font-weight: 600;
    line-height: 1.05;
}
    h2 {
        font-size: 18px; 
        color: #007bff; 
        text-align: center;
        margin-top: 20px;
        margin-bottom: 15px;
        line-height: 1.3;
    }

    .stButton>button {
        background-color: #0492C2; 
        color: black;
        font-size: 1.2em;
        padding: 12px 20px; 
        border-radius: 8px; 
        border: none;
        width: 100%;
        transition: background-color 0.3s ease; 
        line-height: 1.4;
    }
    .stButton>button:hover {
        background-color: #3944BC; 
    }
    .stAlert {
        padding: 18px; 
        border-radius: 10px; 
        font-size: 0.6em;
        line-height: 1.8;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
    }
    .stAlert.success {
        background-color: #d4edda; 
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .stAlert.warning {
        background-color: #fff3cd; 
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .stAlert.error {
        background-color: #f8d7da; 
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

st.write("This application predicts the likelihood of diabetes based on several health parameters. Please input the patient's information below.")

# --- User Input Fields (Main Content) ---
st.subheader("Patient Information")

pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=0)
glucose = st.number_input('Glucose (mg/dL)', min_value=50, max_value=200, value=80)
blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=40, max_value=300, value=70)
skin_thickness = st.number_input('Skin Thickness (mm)', min_value=7, max_value=99, value=20)
insulin = st.number_input('Insulin (mu U/ml)', min_value=15, max_value=846, value=80)
bmi = st.number_input('BMI', min_value=18.0, max_value=67.0, value=21.0, step=0.1, format="%.1f")
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.20, format="%.3f")
age = st.number_input('Age (years)', min_value=21, max_value=110, value=25)
submit_button = st.button('Predict Diabetes')

# --- Prediction Logic ---
if submit_button:
    user_input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                   insulin, bmi, dpf, age]],
                                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Preprocess input
    processed_input = preprocess_input(user_input_df.iloc[0])

    # Feature Engineering
    processed_input['BMI_Category'] = bmi_category(processed_input['BMI'])
    processed_input['glucose_risk'] = glucose_risk(processed_input['Glucose'])
    processed_input['Age_Group'] = age_group(processed_input['Age'])

    # Prepare final input for prediction
    final_input = pd.DataFrame([processed_input])
    final_input = final_input[[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Category',
        'glucose_risk', 'Age_Group'
    ]]

    # Scale the input
    scaled_input = scaler.transform(final_input)

    # Get prediction probability
    prediction_proba = rf_model.predict_proba(scaled_input)[:, 1][0]

    # Display results based on probability thresholds
    st.subheader("Prediction Result:")
    if prediction_proba < 0.3:
        st.success(f"\n\n💚 Low chance of Diabetes. (Probability: {prediction_proba:.2f})")
    elif 0.3 <= prediction_proba < 0.7:
        st.warning(f"\n\n⚠️ Moderate chance of Diabetes. (Probability: {prediction_proba:.2f})")
    else:
        st.error(f"\n\n💔 High chance of Diabetes. (Probability: {prediction_proba:.2f})")

# --- Disclaimer ---
st.markdown("""
<small>Disclaimer: This application provides a prediction based on a machine learning model and should not be considered as medical advice. Always consult with a healthcare professional for diagnosis and treatment.</small>
""", unsafe_allow_html=True)
