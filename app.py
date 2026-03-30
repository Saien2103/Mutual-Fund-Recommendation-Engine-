import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. SET UP THE PAGE
st.set_page_config(page_title="Mutual Fund Advisor", layout="centered")
st.title("🎯 Mutual Fund Recommendation System")
st.write("Enter your details below to find the best mutual funds for you.")


# 2. LOAD THE SAVED MODEL AND ENCODERS
@st.cache_resource  # This keeps the model in memory so the app is fast
def load_assets():
    model = joblib.load('mutual_fund_model.pkl')
    encoders = joblib.load('feature_encoders.pkl')
    target_le = joblib.load('target_encoder.pkl')
    return model, encoders, target_le


model, encoders, target_le = load_assets()

# 3. CREATE THE USER INTERFACE (Frontend)
with st.form("user_input_form"):
    st.header("Your Profile")

    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox("Age Group", encoders['Age Group'].classes_)
        income = st.selectbox("Annual Income", encoders['Min. Income'].classes_)
        investment = st.number_input("Minimum Investment (₹)", min_value=100, value=500)

    with col2:
        duration = st.selectbox("Investment Duration", encoders['Duration'].classes_)
        risk = st.selectbox("Risk Appetite", encoders['Risk'].classes_)
        returns = st.slider("Expected Return (%)", 5.0, 30.0, 12.0)

    # These usually depend on the fund, but we need them for the model
    st.header("Preferences")
    invest_type = st.selectbox("Investment Type", encoders['Investment Type'].classes_)
    sector = st.selectbox("Preferred Sector", encoders['Sector/Industry'].classes_)

    submit = st.form_submit_button("Get Recommendations")

# 4. PREDICTION LOGIC
if submit:
    # Convert text inputs to numbers using the saved encoders
    input_data = pd.DataFrame([[
        encoders['Age Group'].transform([age])[0],
        investment,
        encoders['Duration'].transform([duration])[0],
        encoders['Risk'].transform([risk])[0],
        returns,
        encoders['Investment Type'].transform([invest_type])[0],
        encoders['Sector/Industry'].transform([sector])[0],
        encoders['Min. Income'].transform([income])[0]
    ]], columns=['Age Group', 'Min. Investment', 'Duration', 'Risk', 'Exp. Return',
                 'Investment Type', 'Sector/Industry', 'Min. Income'])

    # Get Top 3 Recommendations
    probs = model.predict_proba(input_data)[0]
    top_3_idx = np.argsort(probs)[-3:][::-1]
    recommendations = target_le.inverse_transform(top_3_idx)

    # Display Results
    st.success("### Top 3 Recommended Funds for You:")
    for i, fund in enumerate(recommendations, 1):
        st.subheader(f"{i}. {fund}")