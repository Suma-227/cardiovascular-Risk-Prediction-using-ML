import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Load the saved model, scaler, and feature names
scaler = joblib.load('models/scaler.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
feature_names = joblib.load('models/feature_names.pkl')
conf_matrix_xgb = joblib.load('models/conf_matrix_xgb.pkl')
feature_importance_xgb = joblib.load('models/feature_importance_xgb.pkl')

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS Styling
st.markdown(
    """
    <style>
    body {background-color: #f0f2f6;}
    .main {background-color: #ffffff; max_width: 50%; margin:auto; border-radius: 12px; padding: 25px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);}
    .stButton>button {background-color: #008CBA; color: white; font-size: 18px; font-weight: bold; border-radius: 6px; padding: 12px; border: none;}
    .stNumberInput input, .stSelectbox>div>div, .stRadio>div {font-size: 1.1rem; border-radius: 5px; border: 1px solid #ccc;}
    h1, h2, h3 {color: #008CBA; font-weight: bold;}
    .large-font {font-size: 26px !important; color: #008CBA; font-weight: bold;}
    .stTabs [role="tab"] {font-size: 18px; font-weight: bold; border-radius: 6px;}
    .stTabs [role="tab"][aria-selected="true"] {background-color: #008CBA; color: white;}
    .footer {position: fixed; bottom: 0; width: 100%; background: #008CBA; color: white; text-align: center; padding: 10px; font-size: 16px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit interface
st.markdown('<h1 style="text-align: center;">💓 Cardiovascular Disease Prediction 💓</h1>', unsafe_allow_html=True)
st.markdown("#### Enter Patient Details:")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    age = st.number_input('Age (years)', min_value=0, max_value=100, value=50)
    height = st.number_input('Height (cm)', min_value=120, max_value=220, value=170)
    weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)

with col2:
    ap_hi = st.number_input('Systolic Blood Pressure', min_value=80, max_value=200, value=120)
    ap_lo = st.number_input('Diastolic Blood Pressure', min_value=60, max_value=140, value=80)

with col3:
    cholesterol = st.selectbox('Cholesterol', options=[1, 2, 3], format_func=lambda x: {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}.get(x))
    gluc = st.selectbox('Glucose', options=[1, 2, 3], format_func=lambda x: {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}.get(x))

with col4:
    smoke = st.radio('Smoking', options=[0, 1], format_func=lambda x: 'Yes' if x else 'No')
    alco = st.radio('Alcohol Intake', options=[0, 1], format_func=lambda x: 'Yes' if x else 'No')

with col5:
    active = st.radio('Physical Activity', options=[0, 1], format_func=lambda x: 'Yes' if x else 'No')

with col6:
    gender = st.radio('Gender', options=[1, 2], format_func=lambda x: 'Female' if x == 1 else 'Male')

input_data = pd.DataFrame({
    'age': [age],
    'height': [height],
    'weight': [weight],
    'ap_hi': [ap_hi],
    'ap_lo': [ap_lo],
    'cholesterol_2': [1 if cholesterol == 2 else 0],
    'cholesterol_3': [1 if cholesterol == 3 else 0],
    'gluc_2': [1 if gluc == 2 else 0],
    'gluc_3': [1 if gluc == 3 else 0],
    'smoke': [smoke],
    'alco': [alco],
    'active': [active],
    'gender_2': [1 if gender == 2 else 0]
})

input_data = input_data[feature_names]

# Prediction and Results Display
if st.button('🔍 Predict with XGBoost'):
    with st.spinner('🔄 Predicting...'):
        import time
        time.sleep(1)  # Add a slight delay to show the loading spinner
        scaled_input_data = scaler.transform(input_data)
        prediction = xgb_model.predict(scaled_input_data)
    
    if prediction[0] == 1:
        st.error('⚠️ Cardiovascular Disease Prediction (XGBoost): **Yes**')
    else:
        st.success('✅ Cardiovascular Disease Prediction (XGBoost): **No**')

# Visualizations
st.subheader('📊 Visualization')
tab1, tab2, tab3 = st.tabs(['Confusion Matrix', 'Feature Importance', 'Accuracy'])

with tab1:
    st.subheader('Confusion Matrix - XGBoost')
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix_xgb, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

with tab2:
    st.subheader('Feature Importance - XGBoost')
    fig, ax = plt.subplots()
    ax.barh(feature_names, feature_importance_xgb, color='#008CBA')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

with tab3:
    st.subheader('Model Accuracy')
    accuracy_xgb = (conf_matrix_xgb[0,0] + conf_matrix_xgb[1,1]) / conf_matrix_xgb.sum()
    st.markdown(f'<p class="large-font">🔹 XGBoost Accuracy: {accuracy_xgb:.2%}</p>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100vw; /* Full viewport width */
        background: #008CBA;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 16px;
        z-index: 100;
    }
    </style>
    <div class="footer">Made with ❤️ by CMRTC | © 2025 All Rights Reserved</div>
    """,
    unsafe_allow_html=True
)
