import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="ShenoySreenivas/tourism-package-model",
        filename="best_tourism_model.pkl",
        repo_type="model",
        force_download=True,
        local_dir="."
    )
    return joblib.load(model_path)


model = load_model()

st.title("Tourism Package Purchase Prediction")

st.write(
    "This tool helps estimate whether a customer is likely to purchase a travel package "
    "based on their profile and engagement details."
)

# Input Form
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    durationpitch = st.slider("Duration of Pitch (minutes)", 0, 60, 15)
    occupation = st.selectbox("Occupation", ["Small Business", "Salaried", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    noperson = st.number_input("Number of People Visiting", 1, 10, 2)
    nofollowups = st.number_input("Number of Follow-ups", 0, 10, 2)
    product = st.selectbox("Product Type", ["Basic", "Standard", "Deluxe", "Super Deluxe"])
    property_star = st.slider("Preferred Hotel Star Rating", 1, 5, 3)
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    notrips = st.slider("Trips per Year", 0, 15, 2)
    passport = st.selectbox("Passport Available", ["No", "Yes"])
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    owncar = st.selectbox("Owns a Car", ["No", "Yes"])
    nochildren = st.number_input("Number of Children Visiting", 0, 5, 0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", 10000, 200000, 30000)

    submit = st.form_submit_button("Predict")

# Encoding values (matches training encoding)
mapping = {
    "Company Invited": 0, "Self Enquiry": 1,
    "Free Lancer": 0, "Small Business": 1, "Salaried": 2,
    "Female": 0, "Male": 1,
    "Basic": 0, "Standard": 1, "Deluxe": 2, "Super Deluxe": 3,
    "Divorced": 0, "Married": 1, "Single": 2,
    "No": 0, "Yes": 1,
    "Executive": 0, "Manager": 1, "Senior Manager": 2, "AVP": 3, "VP": 4
}

# Prepare input row
input_df = pd.DataFrame([[
    age,
    mapping[typeofcontact],
    citytier,
    durationpitch,
    mapping[occupation],
    mapping[gender],
    noperson,
    nofollowups,
    mapping[product],
    property_star,
    mapping[marital],
    notrips,
    mapping[passport],
    pitch_score,
    mapping[owncar],
    nochildren,
    mapping[designation],
    income
]], columns=[
    'Age','TypeofContact','CityTier','DurationOfPitch','Occupation','Gender',
    'NumberOfPersonVisiting','NumberOfFollowups','ProductPitched',
    'PreferredPropertyStar','MaritalStatus','NumberOfTrips','Passport',
    'PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting',
    'Designation','MonthlyIncome'
])

# Prediction
if submit:
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.success(f"The customer is likely to purchase the package.\nEstimated confidence: {probability:.2f}")
    else:
        st.warning(f"The customer is unlikely to purchase the package.\nEstimated confidence: {probability:.2f}")
