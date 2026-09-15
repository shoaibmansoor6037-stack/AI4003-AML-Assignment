
import pandas as pd

# Load Plant 1 data
generation = pd.read_csv("data/Plant_1_Generation_Data.csv")
sensor = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")

# Convert timestamps
generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"])
sensor["DATE_TIME"] = pd.to_datetime(sensor["DATE_TIME"])

print("Generation data:", generation.shape)
print("Sensor data:", sensor.shape)
