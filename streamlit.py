import streamlit as st
import joblib

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

label_map = {
    0: "sadness", 1: "joy", 2: "love",
    3: "anger", 4: "fear", 5: "surprise"
}

st.title("Emotion Detector")

text = st.text_area("Enter your sentence:")

if st.button("Predict Emotion"):
    if text:
        vector = vectorizer.transform([text.lower()])
        prediction = model.predict(vector)[0]
        emotion = label_map[prediction]
        st.success(f"Predicted Emotion: **{emotion.upper()}**")
    else:
        st.error("Please enter some text.")
