import streamlit as st
import pandas as pd
import pickle

# STEP 1: Configure the web application
st.set_page_config(
    page_title="Food Delivery Time Predictor", page_icon="🍔", layout="centered"
)


# STEP 2: Load the trained model
@st.cache_resource
def load_model():
    with open("delivery_model.pkl", "rb") as file:
        package = pickle.load(file)
    return package["model"], package["features"]


model, features = load_model()

# STEP 3: Display the application heading
st.title("🍔 Food Delivery Time Predictor")

st.write("Estimate food delivery time using a trained " "Linear Regression model.")

st.info("Enter the details of a new order to estimate " "its delivery duration.")

# STEP 4: Collect delivery information
st.subheader("Enter Delivery Details")

distance_km = st.number_input(
    "Delivery distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.5
)

preparation_time_min = st.number_input(
    "Food preparation time (minutes)", min_value=1, max_value=120, value=15
)

traffic_level = st.selectbox(
    "Traffic level",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 - Low",
        2: "2 - Moderate",
        3: "3 - High",
        4: "4 - Severe",
    }[x],
)

weather_severity = st.selectbox(
    "Weather severity",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "0 - Clear",
        1: "1 - Light adverse weather",
        2: "2 - Bad weather",
        3: "3 - Severe weather",
    }[x],
)

rider_experience_years = st.number_input(
    "Rider experience (years)", min_value=0, max_value=50, value=3
)

# STEP 5: Predict when the user clicks the button
if st.button("Predict Delivery Time", type="primary"):

    # Prepare the new input
    new_delivery = pd.DataFrame(
        [
            [
                distance_km,
                preparation_time_min,
                traffic_level,
                weather_severity,
                rider_experience_years,
            ]
        ],
        columns=features,
    )

    # Make the prediction
    prediction = model.predict(new_delivery)[0]

    # Prevent a negative displayed duration
    prediction = max(0, prediction)

    # Display the result
    st.success(f"Estimated delivery time: {prediction:.1f} minutes")

    st.caption("This is a model estimate, not a guaranteed arrival time.")

st.divider()

st.caption(
    "Educational project | Python | Pandas | " "Scikit-learn | Pickle | Streamlit"
)
