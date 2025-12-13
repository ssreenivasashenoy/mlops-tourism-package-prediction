import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="ShenoySreenivas/tourism-package-model",
    filename="best_tourism_model.pkl",
    repo_type="model"
)
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction")

st.write(
    """
    This application predicts whether a customer is likely to purchase
    the Wellness Tourism Package based on customer and interaction details.
    """
)

# User inputs
Age = st.number_input("Age", 18, 80, 30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", 0, 60, 15)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Small Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", 1, 10, 2)
NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 2)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
NumberOfTrips = st.number_input("Number of Trips Per Year", 0, 20, 2)
Passport = st.selectbox("Passport Available", ["No", "Yes"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own Car", ["No", "Yes"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", 0, 5, 0)
Designation = st.selectbox(
    "Designation",
    ["AVP", "Executive", "Manager", "Senior Manager", "VP"]
)
MonthlyIncome = st.number_input("Monthly Income", 10000, 200000, 30000)

# Encoding (EXACT match with LabelEncoder used during training)
encoding = {
    "TypeofContact": {
        "Company Invited": 0,
        "Self Enquiry": 1
    },
    "Occupation": {
        "Free Lancer": 0,
        "Salaried": 1,
        "Small Business": 2
    },
    "Gender": {
        "Female": 0,
        "Male": 1
    },
    "ProductPitched": {
        "Basic": 0,
        "Deluxe": 1,
        "Standard": 2,
        "Super Deluxe": 3
    },
    "MaritalStatus": {
        "Divorced": 0,
        "Married": 1,
        "Single": 2
    },
    "Passport": {
        "No": 0,
        "Yes": 1
    },
    "OwnCar": {
        "No": 0,
        "Yes": 1
    },
    "Designation": {
        "AVP": 0,
        "Executive": 1,
        "Manager": 2,
        "Senior Manager": 3,
        "VP": 4
    }
}

# Prepare input DataFrame (same order as training)
input_df = pd.DataFrame([[ 
    Age,
    encoding["TypeofContact"][TypeofContact],
    CityTier,
    DurationOfPitch,
    encoding["Occupation"][Occupation],
    encoding["Gender"][Gender],
    NumberOfPersonVisiting,
    NumberOfFollowups,
    encoding["ProductPitched"][ProductPitched],
    PreferredPropertyStar,
    encoding["MaritalStatus"][MaritalStatus],
    NumberOfTrips,
    encoding["Passport"][Passport],
    PitchSatisfactionScore,
    encoding["OwnCar"][OwnCar],
    NumberOfChildrenVisiting,
    encoding["Designation"][Designation],
    MonthlyIncome
]], columns=[
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Likely to Purchase")
    else:
        st.error("Unlikely to Purchase")

    st.write(f"Confidence Score: **{probability:.2f}**")
