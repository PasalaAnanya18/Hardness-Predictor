import joblib
import streamlit as st
import numpy as np
import time

model=joblib.load('xgboost_hardness_model.pkl')

st.title("Cu-Ni-Si Alloy Hardness Predictor")
st.markdown("---")

col1,col2=st.columns(2)

with col1:
    st.header("Alloy Composition(%)")
    cu=st.number_input('Copper(%)', min_value=0.0, max_value=100.0,value=90.0)
    ni = st.number_input('Nickel (%)', min_value=0.0, max_value=100.0, value=5.0)
    si = st.number_input('Silicon (%)', min_value=0.0, max_value=100.0, value=2.0)

with col2:   
    st.header("Other Elements(%)")
    al = st.number_input('Aluminum (%)', min_value=0.0, max_value=10.0, value=0.5)  # Typical trace amount
    mg = st.number_input('Magnesium (%)', min_value=0.0, max_value=5.0, value=0.2)

st.header("Processing Conditions")
col1,col2=st.columns(2)
with col1:
    solution_temp = st.number_input('Solution Temperature (°C)', min_value=0, max_value=1200, value=800)
    tss = st.number_input('Tss (K)', min_value=0.0, max_value=2000.0, value=1073.0)
    
with col2:
    aging = st.selectbox('Aging Performed?', ['Yes', 'No'])
    aging_numeric = 1 if aging == 'Yes' else 0

st.header("Measured Properties")
uts = st.number_input('Ultimate Tensile Strength (MPa)', min_value=0.0, value=500.0)
ys = st.number_input('Yield Strength (MPa)', min_value=0.0, value=400.0)

default_features = {
    'Co': 0.1, 
    'CR reduction (%)': 30.0,
    'Electrical conductivity (%IACS)': 50.0,
    'Cr': 0.05,
    'tag (h)': 1.0,
    'tss (h)': 1.0,
    'Zn': 0.1,
    'Sn': 0.05,
    'Secondary thermo-mechanical process': 0,
    'Tag (K)': 1073.0
}

features = np.array([[
    uts, ys, tss, al, cu, 
    ni, default_features['Co'], default_features['CR reduction (%)'],
    default_features['Electrical conductivity (%IACS)'], default_features['Cr'],
    default_features['tag (h)'], default_features['tss (h)'], default_features['Zn'],
    si, default_features['Sn'], aging_numeric,
    solution_temp, default_features['Secondary thermo-mechanical process'],default_features['Tag (K)']
]])

if st.button('Predict Hardness',type='primary'):
    with st.spinner('Calculating hardness...'):
        time.sleep(0.5)
        prediction = model.predict(features)
    st.markdown(
        f"<div style='text-align:center; font-size:48px; font-weight:bold; color:#2E86C1;'>"
        f"Predicted Hardness: {prediction[0]:.2f} HV"
        f"</div>",
        unsafe_allow_html=True
    )
   
    min_hardness = 142
    max_hardness = 343
    progress = int(100 * (prediction[0] - min_hardness) / (max_hardness - min_hardness))
    progress = max(0, min(progress, 100))
    st.progress(progress)

   