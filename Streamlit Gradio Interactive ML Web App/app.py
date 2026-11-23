
import streamlit as st
import pandas as pd
import joblib

# Feature order used during model training
FEATURES = ["pclass", "sex", "age", "fare"]

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


# App title
st.title("Titanic Survival Prediction")

st.write(
    "Predict passenger survival using a Machine Learning model."
)

st.divider()

# Passenger details
st.subheader("Enter Passenger Details")

# Ticket class
pclass = st.selectbox(
    "Ticket Class",
    [1, 2, 3]
)

# Gender
sex = st.radio(
    "Gender",
    ["Female", "Male"],
    horizontal=True
)

# Age
age = st.slider(
    "Passenger Age",
    min_value=1,
    max_value=80,
    value=30
)

# Fare
fare = st.slider(
    "Ticket Fare",
    min_value=0,
    max_value=500,
    value=30
)

# Prediction button
if st.button("Predict Survival"):

    try:
        model = load_model()

        # Convert gender into 0 and 1
        gender = 1 if sex == "Female" else 0

        # Create input DataFrame
        input_data = pd.DataFrame(
            [[pclass, gender, age, fare]],
            columns=FEATURES
        )

        # Predict survival probability
        chance = model.predict_proba(input_data)[0][1]

        # Display prediction
        st.subheader("Prediction Result")

        st.metric(
            "Estimated Survival Chance",
            f"{chance:.1%}"
        )

        if chance >= 0.5:
            st.success("The model predicts: Likely to Survive!")
        else:
            st.error("The model predicts: Unlikely to Survive!")

    except FileNotFoundError:
        st.error(
            "Model file is missing. Please add model.joblib."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# Educational note
st.divider()

st.caption(
    "This educational model was trained using the historical "
    "Titanic dataset. Predictions are estimates, not advice."
)
