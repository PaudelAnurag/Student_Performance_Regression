from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# =====================================
# Load Models
# =====================================

MODEL_DIR = Path("../model")

if not MODEL_DIR.exists():
    MODEL_DIR = Path("model")

model = joblib.load(MODEL_DIR / "best_model.pkl")
label_encoders = joblib.load(MODEL_DIR /"label_encoders.pkl")

# ==============================
# Title
# ==============================

st.set_page_config(page_title="Student Performance Prediction")

st.title(" Student Performance Prediction")
st.write("Predict a student's final grade (G3).")

# ==============================
# User Inputs
# ==============================

school = st.selectbox("School", ["GP", "MS"])

sex = st.selectbox("Sex", ["F", "M"])

age = st.slider("Age", 15, 22, 18)

address = st.selectbox("Address", ["U", "R"])

famsize = st.selectbox("Family Size", ["GT3", "LE3"])

Pstatus = st.selectbox("Parent Status", ["A", "T"])

Medu = st.selectbox("Mother Education", [0,1,2,3,4])

Fedu = st.selectbox("Father Education", [0,1,2,3,4])

Mjob = st.selectbox(
    "Mother Job",
    ["teacher","health","services","at_home","other"]
)

Fjob = st.selectbox(
    "Father Job",
    ["teacher","health","services","at_home","other"]
)

reason = st.selectbox(
    "Reason for choosing school",
    ["course","home","reputation","other"]
)

guardian = st.selectbox(
    "Guardian",
    ["mother","father","other"]
)

traveltime = st.selectbox("Travel Time", [1,2,3,4])

studytime = st.selectbox("Study Time", [1,2,3,4])

failures = st.selectbox("Past Class Failures", [0,1,2,3])

schoolsup = st.selectbox("School Support", ["yes","no"])

famsup = st.selectbox("Family Support", ["yes","no"])

paid = st.selectbox("Extra Paid Classes", ["yes","no"])

activities = st.selectbox("Extra Activities", ["yes","no"])

nursery = st.selectbox("Attended Nursery", ["yes","no"])

higher = st.selectbox("Wants Higher Education", ["yes","no"])

internet = st.selectbox("Internet Access", ["yes","no"])

romantic = st.selectbox("In Relationship", ["yes","no"])

famrel = st.slider("Family Relationship",1,5,4)

freetime = st.slider("Free Time",1,5,3)

goout = st.slider("Go Out",1,5,3)

Dalc = st.slider("Weekday Alcohol",1,5,1)

Walc = st.slider("Weekend Alcohol",1,5,1)

health = st.slider("Health",1,5,3)

absences = st.number_input("Absences",0,100,0)

G1 = st.slider("First Period Grade (G1)",0,20,10)

G2 = st.slider("Second Period Grade (G2)",0,20,10)

# ==============================
# Prediction
# ==============================

if st.button("Predict G3"):

    # Label Encoding

    school = label_encoders["school"].transform([school])[0]

    sex = label_encoders["sex"].transform([sex])[0]

    address = label_encoders["address"].transform([address])[0]

    famsize = label_encoders["famsize"].transform([famsize])[0]

    Pstatus = label_encoders["Pstatus"].transform([Pstatus])[0]

    guardian = label_encoders["guardian"].transform([guardian])[0]

    schoolsup = label_encoders["schoolsup"].transform([schoolsup])[0]

    famsup = label_encoders["famsup"].transform([famsup])[0]

    paid = label_encoders["paid"].transform([paid])[0]

    activities = label_encoders["activities"].transform([activities])[0]

    nursery = label_encoders["nursery"].transform([nursery])[0]

    higher = label_encoders["higher"].transform([higher])[0]

    internet = label_encoders["internet"].transform([internet])[0]

    romantic = label_encoders["romantic"].transform([romantic])[0]

    # Feature Engineering

    grade_improvement = G2 - G1

    study_efficiency = studytime / (absences + 1)

    risk = (
        failures +
        absences / 10 +
        goout / 5
    )

    # DataFrame

    sample = pd.DataFrame({

        "school":[school],
        "sex":[sex],
        "age":[age],
        "address":[address],
        "famsize":[famsize],
        "Pstatus":[Pstatus],
        "Medu":[Medu],
        "Fedu":[Fedu],
        "Mjob":[Mjob],
        "Fjob":[Fjob],
        "reason":[reason],
        "guardian":[guardian],
        "traveltime":[traveltime],
        "studytime":[studytime],
        "failures":[failures],
        "schoolsup":[schoolsup],
        "famsup":[famsup],
        "paid":[paid],
        "activities":[activities],
        "nursery":[nursery],
        "higher":[higher],
        "internet":[internet],
        "romantic":[romantic],
        "famrel":[famrel],
        "freetime":[freetime],
        "goout":[goout],
        "Dalc":[Dalc],
        "Walc":[Walc],
        "health":[health],
        "absences":[absences],
        "G1":[G1],
        "G2":[G2],
        "grade_improvement":[grade_improvement],
        "study_efficiency":[study_efficiency],
        "risk":[risk]

    })

    prediction = model.predict(sample)[0]

    prediction = max(0, min(20, prediction))

    st.success(f" Predicted Final Grade (G3): **{prediction:.2f}/20**")