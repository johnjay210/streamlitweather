import kagglehub
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load dataset
path = kagglehub.dataset_download(
    "sudalairajkumar/daily-temperature-of-major-cities"
)

file = os.path.join(path, "city_temperature.csv")
df = pd.read_csv(file, low_memory=False)

# Clean data
df = df[df["AvgTemperature"] != -99]

# Feature engineering
df["Date"] = pd.to_datetime(df[["Year","Month","Day"]])
df["DayOfYear"] = df["Date"].dt.dayofyear

# Sample
sample_df = df.sample(n=300000, random_state=42)

X = sample_df[
    ["Region","Country","State","City","Month","Day","Year","DayOfYear"]
].copy()

y = sample_df["AvgTemperature"]

# Encode
encoders = {}
for col in ["Region","Country","State","City"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Train-test split
print("Train model started")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

lr = LinearRegression()
lr.fit(X_train, y_train)

# SAVE MODELS (IMPORTANT FIX)
os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/rf.pkl")
joblib.dump(xgb_model, "models/xgb.pkl")
joblib.dump(lr, "models/lr.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("Models saved successfully!")