import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set page config for a premium layout
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for rich aesthetics (sleek dark/modern theme, glassmorphism cards, and premium styling)
st.markdown("""
<style>
    /* Styling the main container */
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #11111d 100%);
        color: #e0e0e0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Header styling */
    .title-text {
        background: linear-gradient(90deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .subtitle-text {
        color: #a0a0b0;
        font-size: 1.1rem;
        margin-bottom: 30px;
        text-align: center;
    }
    
    /* Result styling */
    .result-box {
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        margin-top: 20px;
    }
    .low-stress {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }
    .high-stress {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    
    /* Input reference table */
    .ref-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 0.9rem;
    }
    .ref-table th, .ref-table td {
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 8px 12px;
        text-align: left;
    }
    .ref-table th {
        background-color: rgba(255, 255, 255, 0.08);
        color: #feb47b;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">🎓 Student Lifestyle & Stress Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Predict student stress levels based on lifestyle indicators using Machine Learning</div>', unsafe_allow_html=True)

# Helper functions to load scaler and model
@st.cache_resource
def load_assets():
    scaler_path = "scaler.pkl"
    model_path = "model.pkl"
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        return None, None
        
    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        return scaler, model
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

scaler, model = load_assets()

if scaler is None or model is None:
    st.error("⚠️ Pre-trained model or scaler files are missing! Please ensure both `scaler.pkl` and `model.pkl` exist in the workspace.")
else:
    # Feature Reference Information
    with st.expander("ℹ️ Feature Reference Table", expanded=True):
        st.markdown("""
        Below is the details of the 7 features in the exact expected order:
        <table class="ref-table">
            <tr>
                <th>Index</th>
                <th>Feature Name</th>
                <th>Description / Expected Ranges</th>
            </tr>
            <tr>
                <td>1</td>
                <td>Student Type</td>
                <td>0 = School, 1 = College, 2 = Working Student</td>
            </tr>
            <tr>
                <td>2</td>
                <td>Sleep Hours</td>
                <td>Typical range: 0.0 to 24.0 hours per day</td>
            </tr>
            <tr>
                <td>3</td>
                <td>Study Hours</td>
                <td>Typical range: 0.0 to 24.0 hours per day</td>
            </tr>
            <tr>
                <td>4</td>
                <td>Social Media Hours</td>
                <td>Typical range: 0.0 to 24.0 hours per day</td>
            </tr>
            <tr>
                <td>5</td>
                <td>Attendance (%)</td>
                <td>Typical range: 0.0% to 100.0%</td>
            </tr>
            <tr>
                <td>6</td>
                <td>Exam Pressure</td>
                <td>Subjective score: 0 (No pressure) to 10 (Highest pressure)</td>
            </tr>
            <tr>
                <td>7</td>
                <td>Family Support</td>
                <td>Subjective score: 0 (No support) to 10 (Highest support)</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

    # Tabs for different inputs
    tab1, tab2 = st.tabs(["💬 Comma-Separated Input", "🎛️ Interactive Form Input"])

    with tab1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Fast Paste Input")
        st.write("Enter 7 values separated by commas in the text box below.")
        
        comma_input = st.text_input(
            label="Input values:",
            value="1, 6.8, 5.2, 3.0, 85.5, 6.0, 7.0",
            placeholder="e.g. 1, 6.8, 5.2, 3.0, 85.5, 6.0, 7.0",
            help="Order: Student_Type, Sleep_Hours, Study_Hours, Social_Media_Hours, Attendance, Exam_Pressure, Family_Support"
        )
        
        predict_tab1 = st.button("Run Prediction (Tab 1)", key="pred_tab1")
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_tab1:
            # 1. Parse input
            try:
                parts = [p.strip() for p in comma_input.split(",")]
                if len(parts) != 7:
                    st.error(f"❌ Input validation failed: Expected exactly 7 values, but got {len(parts)} values instead.")
                else:
                    # Convert to floats
                    float_vals = [float(val) for val in parts]
                    
                    # 2. Detailed validations
                    # Student Type check
                    student_type = float_vals[0]
                    if student_type not in [0.0, 1.0, 2.0]:
                        st.warning("⚠️ Warning: Student Type (first value) should be either 0 (School), 1 (College), or 2 (Working Student).")
                    
                    # Sleep/Study/Social Media check
                    if any(v < 0 or v > 24 for v in float_vals[1:4]):
                        st.warning("⚠️ Warning: Hours values (Sleep, Study, Social Media) should realistically be between 0 and 24.")
                        
                    # Attendance check
                    if float_vals[4] < 0 or float_vals[4] > 100:
                        st.warning("⚠️ Warning: Attendance percentage should be between 0 and 100.")
                        
                    # Pressure/Support scores check
                    if any(v < 0 or v > 10 for v in float_vals[5:7]):
                        st.warning("⚠️ Warning: Exam Pressure and Family Support scores should typically be between 0 and 10.")
                    
                    # 3. Predict
                    input_array = np.array([float_vals])
                    input_scaled = scaler.transform(input_array)
                    
                    prediction = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0]
                    
                    # 4. Display result
                    st.subheader("Prediction Result")
                    if prediction == 1:
                        st.markdown(f'<div class="result-box high-stress">⚠️ HIGH STRESS LEVEL DETECTED (Probability: {prob[1]:.2%})</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box low-stress">✅ LOW / NORMAL STRESS LEVEL DETECTED (Probability: {prob[0]:.2%})</div>', unsafe_allow_html=True)
                        
            except ValueError as ve:
                st.error("❌ Conversion failed: Please ensure all entered values are valid numerical representations (floats or integers).")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")

    with tab2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Interactive Feature Form")
        
        # User Friendly controls
        f_student_type = st.selectbox(
            "Student Type",
            options=["School (0)", "College (1)", "Working Student (2)"],
            index=1
        )
        # Extract the integer mapping value
        student_type_val = float(f_student_type.split("(")[1][0])
        
        f_sleep_hours = st.slider("Sleep Hours (per day)", 0.0, 24.0, 7.0, 0.1)
        f_study_hours = st.slider("Study Hours (per day)", 0.0, 24.0, 4.0, 0.1)
        f_social_media = st.slider("Social Media Hours (per day)", 0.0, 24.0, 3.0, 0.1)
        f_attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0, 0.5)
        f_exam_pressure = st.slider("Exam Pressure Score (0-10)", 0.0, 10.0, 5.0, 0.5)
        f_family_support = st.slider("Family Support Score (0-10)", 0.0, 10.0, 7.0, 0.5)
        
        predict_tab2 = st.button("Run Prediction (Tab 2)", key="pred_tab2")
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_tab2:
            try:
                # Arrange inputs in the exact expected order
                inputs = [
                    student_type_val,
                    f_sleep_hours,
                    f_study_hours,
                    f_social_media,
                    f_attendance,
                    f_exam_pressure,
                    f_family_support
                ]
                
                input_array = np.array([inputs])
                input_scaled = scaler.transform(input_array)
                
                prediction = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0]
                
                # Render results nicely
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.markdown(f'<div class="result-box high-stress">⚠️ HIGH STRESS LEVEL DETECTED (Probability: {prob[1]:.2%})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box low-stress">✅ LOW / NORMAL STRESS LEVEL DETECTED (Probability: {prob[0]:.2%})</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")
