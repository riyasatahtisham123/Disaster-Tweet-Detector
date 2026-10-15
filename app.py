import streamlit as st
import joblib

# Load Model and Count Vectorizer
mnb = joblib.load('alertsphere_model.pkl')
cv = joblib.load('count_vectorizer.pkl')

st.set_page_config(page_title="AlertSphere", page_icon="🚨", layout="centered")

st.title("🚨 Disaster Tweet Detector")
st.write("Enter a tweet below to check whether it's a disaster-related tweet or a normal tweet.")

# Input text box
tweet_text = st.text_area("Enter Tweet:", placeholder="Type your tweet here...")

if st.button("Detect", type="primary"):
    if not tweet_text.strip():
        st.warning("Please enter a tweet first.")
    else:
        # Transform and predict
        new_data_cv = cv.transform([tweet_text])
        pred_mnb = mnb.predict(new_data_cv)
        
        if pred_mnb == 1:
            st.error("Prediction Result: **Disaster Tweet** ⚠️")
        else:
            st.success("Prediction Result: **Normal Tweet** ✅")