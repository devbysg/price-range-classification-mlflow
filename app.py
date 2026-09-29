import streamlit as st

from backend.predictor import predict_price_range


st.set_page_config(
    page_title="CodeX Beverage: Price Prediction",
    page_icon="🥤",
    layout="wide"
)


st.title("CodeX Beverage: Price Prediction")


col1, col2, col3, col4 = st.columns(4)


with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30,
        step=1
    )

    income_level = st.selectbox(
        "Income Level (In L)",
        [
            "<10L",
            "10L - 15L",
            "16L - 25L",
            "26L - 35L",
            ">35L"
        ]
    )

    awareness = st.selectbox(
        "Awareness of other brands",
        [
            "0 to 1",
            "2 to 4",
            "above 4"
        ]
    )

    packaging = st.selectbox(
        "Packaging Preference",
        [
            "Simple",
            "Premium",
            "Eco-Friendly"
        ]
    )


with col2:
    gender = st.selectbox(
        "Gender",
        ["M", "F"]
    )

    consume_frequency = st.selectbox(
        "Consume Frequency(weekly)",
        [
            "0-2 times",
            "3-4 times",
            "5-7 times"
        ]
    )

    reason = st.selectbox(
        "Reason for choosing brands",
        [
            "Price",
            "Quality",
            "Availability",
            "Brand Reputation"
        ]
    )

    health_concerns = st.selectbox(
        "Health Concerns",
        [
            "Low (Not very concerned)",
            "Medium (Moderately health-conscious)",
            "High (Very health-conscious)"
        ]
    )


with col3:
    zone = st.selectbox(
        "Zone",
        [
            "Urban",
            "Metro",
            "Rural",
            "Semi-Urban"
        ]
    )

    current_brand = st.selectbox(
        "Current Brand",
        [
            "Newcomer",
            "Established"
        ]
    )

    flavor = st.selectbox(
        "Flavor Preference",
        [
            "Traditional",
            "Exotic"
        ]
    )

    consumption_situation = st.selectbox(
        "Typical Consumption Situations",
        [
            "Active (eg. Sports, gym)",
            "Casual (eg. At home)",
            "Social (eg. Parties)"
        ]
    )


with col4:
    occupation = st.selectbox(
        "Occupation",
        [
            "Working Professional",
            "Student",
            "Entrepreneur",
            "Retired"
        ]
    )

    consumption_size = st.selectbox(
        "Preferable Consumption Size",
        [
            "Small (250 ml)",
            "Medium (500 ml)",
            "Large (1 L)"
        ]
    )

    purchase_channel = st.selectbox(
        "Purchase Channel",
        [
            "Online",
            "Retail Store"
        ]
    )


st.divider()


if st.button("Calculate Price Range", type="primary"):

    input_data = {
        "age": age,
        "gender": gender,
        "zone": zone,
        "occupation": occupation,
        "income_levels": income_level,
        "consume_frequency(weekly)": consume_frequency,
        "current_brand": current_brand,
        "preferable_consumption_size": consumption_size,
        "awareness_of_other_brands": awareness,
        "reasons_for_choosing_brands": reason,
        "flavor_preference": flavor,
        "purchase_channel": purchase_channel,
        "packaging_preference": packaging,
        "health_concerns": health_concerns,
        "typical_consumption_situations": consumption_situation
    }

    prediction = predict_price_range(input_data)

    st.success(f"Price Range: {prediction}")