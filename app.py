import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import kagglehub
import os
from datetime import datetime

st.set_page_config(
    page_title="Temperature Prediction",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.big-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.metric-card {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    color: white;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
}
.title-text {
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    color: gray;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">🌍 Modern Temperature Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict temperature by city, compare models, and view historical climate trends.</p>', unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    path = kagglehub.dataset_download("sudalairajkumar/daily-temperature-of-major-cities")
    file = os.path.join(path, "city_temperature.csv")

    df = pd.read_csv(file, low_memory=False)
    df = df[df["AvgTemperature"] != -99]

    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df

weather = load_dataset()

lookup = weather[["Region", "Country", "State", "City"]].drop_duplicates()

@st.cache_resource
def load_models():
    models = {
        "Random Forest": joblib.load("models/rf.pkl"),
        "XGBoost": joblib.load("models/xgb.pkl"),
        "Linear Regression": joblib.load("models/lr.pkl")
    }

    encoders = joblib.load("models/encoders.pkl")

    return models, encoders

models, encoders = load_models()

@st.cache_data
def get_regions():
    return sorted(lookup["Region"].dropna().unique())

@st.cache_data
def get_countries(region):
    return sorted(lookup[lookup["Region"] == region]["Country"].dropna().unique())

@st.cache_data
def get_cities(country):
    return sorted(lookup[lookup["Country"] == country]["City"].dropna().unique())

@st.cache_data
def get_state(country, city):
    values = lookup[
        (lookup["Country"] == country) &
        (lookup["City"] == city)
    ]["State"].dropna().unique()

    return values[0] if len(values) > 0 else ""

@st.cache_data
def get_city_history(country, city):
    return weather[
        (weather["Country"] == country) &
        (weather["City"] == city)
    ].sort_values("Date")

def encode_input(input_df):
    for col in ["Region", "Country", "State", "City"]:
        mapping = {
            label: idx
            for idx, label in enumerate(encoders[col].classes_)
        }

        input_df[col] = (
            input_df[col]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

    return input_df

st.sidebar.title("⚙️ Settings")

selected_model = st.sidebar.selectbox(
    "Choose Prediction Model",
    list(models.keys())
)

unit = st.sidebar.radio(
    "Temperature Unit",
    ["Fahrenheit", "Celsius"]
)

st.sidebar.info("""
Models available:
- Random Forest
- XGBoost
- Linear Regression
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "📈 Historical Trend",
    "📊 Monthly Analysis",
    "ℹ️ About"
])

with tab1:
    st.markdown('<div class="big-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        region = st.selectbox("Region", get_regions())

    with col2:
        country = st.selectbox("Country", get_countries(region))

    with col3:
        city = st.selectbox("City", get_cities(country))

    state = get_state(country, city)

    st.write("📍 State:", state if state else "")

    col4, col5, col6 = st.columns(3)

    with col4:
        month = st.slider("Month", 1, 12, 1)

    with col5:
        day = st.slider("Day", 1, 31, 1)

    with col6:
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2026)

    predict_btn = st.button("🚀 Predict Temperature", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        try:
            date_value = pd.to_datetime(
                f"{int(year)}-{int(month)}-{int(day)}",
                errors="raise"
            )

            input_df = pd.DataFrame({
                "Region": [region],
                "Country": [country],
                "State": [state],
                "City": [city],
                "Month": [month],
                "Day": [day],
                "Year": [year],
                "DayOfYear": [date_value.dayofyear]
            })

            input_df = encode_input(input_df)

            prediction = models[selected_model].predict(input_df)[0]
            celsius = (prediction - 32) * 5 / 9

            if unit == "Celsius":
                display_temp = celsius
                symbol = "°C"
            else:
                display_temp = prediction
                symbol = "°F"

            colA, colB, colC = st.columns(3)

            with colA:
                st.metric("🌡️ Predicted Temperature", f"{display_temp:.2f} {symbol}")

            with colB:
                st.metric("🤖 Model Used", selected_model)

            with colC:
                st.metric("📅 Day of Year", date_value.dayofyear)

            report = pd.DataFrame({
                "Region": [region],
                "Country": [country],
                "State": [state],
                "City": [city],
                "Date": [date_value.date()],
                "Model": [selected_model],
                "Temperature_F": [round(prediction, 2)],
                "Temperature_C": [round(celsius, 2)]
            })

            csv = report.to_csv(index=False)

            st.download_button(
                "⬇️ Download Prediction Report",
                csv,
                "temperature_prediction_report.csv",
                "text/csv",
                use_container_width=True
            )

        except Exception:
            st.error("Invalid date selected. Please check the month and day.")

with tab2:
    st.subheader(f"📈 Historical Temperature Trend")

    city_data = get_city_history(country, city)

    if not city_data.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(city_data["Date"], city_data["AvgTemperature"])
        ax.set_xlabel("Year")
        ax.set_ylabel("Temperature (°F)")
        ax.set_title(f"Historical Temperature Trend - {city}")
        st.pyplot(fig)

        st.write("Highest recorded temperature:", round(city_data["AvgTemperature"].max(), 2), "°F")
        st.write("Lowest recorded temperature:", round(city_data["AvgTemperature"].min(), 2), "°F")
        st.write("Average temperature:", round(city_data["AvgTemperature"].mean(), 2), "°F")
    else:
        st.warning("No historical data found for this city.")

with tab3:
    st.subheader("📊 Average Monthly Temperature")

    city_data = get_city_history(country, city)

    if not city_data.empty:
        monthly = city_data.groupby("Month")["AvgTemperature"].mean()

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(monthly.index, monthly.values, marker="o")
        ax2.set_xticks(range(1, 13))
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Average Temperature (°F)")
        ax2.set_title(f"Monthly Average Temperature - {city}")
        st.pyplot(fig2)

        st.dataframe(monthly.reset_index().rename(
            columns={"AvgTemperature": "Average Temperature"}
        ))
    else:
        st.warning("No monthly data available.")

with tab4:
    st.subheader("ℹ️ About This System")

    st.write("""
    This system predicts average daily temperature using machine learning.

    It uses:
    - Region
    - Country
    - State
    - City
    - Month
    - Day
    - Year
    - Day of Year

    The system supports three models:
    - Random Forest
    - XGBoost
    - Linear Regression

    It also displays historical trends and monthly temperature analysis.
    """)