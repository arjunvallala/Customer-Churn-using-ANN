import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction")
st.write("Enter the customer's details below.")

# Customer inputs
tenure_months = st.number_input(
    "Tenure (months)", min_value=1, max_value=72, value=12
)

monthly_charges = st.number_input(
    "Monthly Charges", min_value=150.0, max_value=1500.0, value=750.0
)

support_calls = st.number_input(
    "Support Calls", min_value=0, max_value=10, value=2
)

payment_method = st.selectbox(
    "Payment Method",
    ["Bank transfer", "Cash", "Credit card", "Debit card", "UPI"]
)

contract_type = st.selectbox(
    "Contract Type",
    ["Monthly", "One year", "Two year"]
)

# Display customer dictionary
customer = {
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "support_calls": support_calls,
    "payment_method": payment_method,
    "contract_type": contract_type
}

st.subheader("Customer Dictionary")
st.json(customer)

# Load trained preprocessing objects and ANN
@st.cache_resource
def load_artifacts():
    with open("ohe.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    ann = load_model("ann.keras")

    return encoder, scaler, ann


try:
    ohe, scaler, model = load_artifacts()

    # Convert dictionary into a single-row DataFrame
    input_data = pd.DataFrame([customer])

    # IMPORTANT: use the same order as encoder.fit()
    categorical_data = input_data[
        ["payment_method", "contract_type"]
    ]

    encoded = ohe.transform(categorical_data)

    encoded_data = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(),
        index=input_data.index
    )

    # Combine numerical and encoded features
    processed_data = pd.concat(
        [
            input_data[
                ["tenure_months", "monthly_charges", "support_calls"]
            ],
            encoded_data
        ],
        axis=1
    )

    # Match the exact feature order used to train the scaler
    if hasattr(scaler, "feature_names_in_"):
        processed_data = processed_data[
            list(scaler.feature_names_in_)
        ]

    # Apply existing preprocessing; never refit on one customer
    scaled_data = scaler.transform(processed_data)

    # Predict churn probability
    probability = float(
        np.asarray(model.predict(scaled_data, verbose=0)).reshape(-1)[0]
    )

    st.subheader("Prediction")
    st.metric("Churn Probability", f"{probability:.2%}")

    if probability >= 0.5:
        st.error("Prediction: Customer is likely to churn.")
    else:
        st.success("Prediction: Customer is likely to stay.")

except Exception as e:
    st.error(f"Could not generate prediction: {e}")
