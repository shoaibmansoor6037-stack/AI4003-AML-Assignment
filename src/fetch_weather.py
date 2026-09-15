
import requests
import pandas as pd

# Open-Meteo archive API
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 14.82,
    "longitude": 78.28,
    "start_date": "2020-05-15",
    "end_date": "2020-06-17",
    "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
    "timezone": "Asia/Kolkata"
}

# Request weather data
response = requests.get(url, params=params)

print("API status:", response.status_code)

# Convert response to DataFrame
data = response.json()

weather = pd.DataFrame(data["hourly"])

# Rename columns
weather = weather.rename(columns={
    "time": "datetime",
    "shortwave_radiation": "sw_radiation",
    "temperature_2m": "temp_2m",
    "cloud_cover": "cloud_cover"
})

# Convert datetime
weather["datetime"] = pd.to_datetime(weather["datetime"])

# Save data
weather.to_csv(
    "data/plant1_openmeteo.csv",
    index=False
)

print("Open-Meteo data saved successfully!")
