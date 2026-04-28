import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Student Dropout Predictor")

income = st.number_input("Family Income")
study = st.number_input("Study Hours per Day")
stress = st.number_input("Stress Index")

education = st.selectbox("Parental Education", ["High School", "UG", "PG"])

# SAME encoding as training
edu_map = {"High School": 0, "UG": 1, "PG": 2}
edu_val = edu_map[education]

if st.button("Predict"):
    input_data = np.array([[income, study, stress, edu_val]])
    result = model.predict(input_data)

    if result[0] == 1:
        st.error("High chance of Dropout")
    else:
        st.success("Low chance of Dropout")