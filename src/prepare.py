
import pandas as pd
import numpy as np

# Load prepared hourly data
hourly = pd.read_csv("data/plant1_hourly.csv")
weather = pd.read_csv("data/plant1_openmeteo.csv")

hourly["datetime"] = pd.to_datetime(hourly["datetime"])
weather["datetime"] = pd.to_datetime(weather["datetime"])

# Merge with Open-Meteo data
comparison = hourly.merge(
    weather,
    on="datetime",
    how="inner"
)

# Convert radiation to kW/m²
comparison["sw_radiation_kw"] = (
    comparison["sw_radiation"] / 1000
)

# Extract hour
comparison["hour"] = comparison["datetime"].dt.hour

# Train: May 15 to June 10
train = comparison[
    (comparison["datetime"] >= "2020-05-15") &
    (comparison["datetime"] < "2020-06-11")
].copy()

# Test: June 11 to June 17
test = comparison[
    (comparison["datetime"] >= "2020-06-11") &
    (comparison["datetime"] < "2020-06-18")
].copy()

# Remove missing values required by the model
train = train.dropna(
    subset=[
        "ac_power",
        "irradiation",
        "module_temp",
        "ambient_temp"
    ]
).copy()

test = test.dropna(
    subset=[
        "ac_power",
        "irradiation",
        "module_temp",
        "ambient_temp"
    ]
).copy()

# Encode hour of day
train["sin_hour"] = np.sin(
    2 * np.pi * train["hour"] / 24
)

train["cos_hour"] = np.cos(
    2 * np.pi * train["hour"] / 24
)

test["sin_hour"] = np.sin(
    2 * np.pi * test["hour"] / 24
)

test["cos_hour"] = np.cos(
    2 * np.pi * test["hour"] / 24
)

print("Training rows:", len(train))
print("Testing rows:", len(test))
