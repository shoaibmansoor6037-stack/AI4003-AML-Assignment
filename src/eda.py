
import pandas as pd
import matplotlib.pyplot as plt

# Load hourly data
hourly = pd.read_csv("data/plant1_hourly.csv")

# Convert datetime
hourly["datetime"] = pd.to_datetime(hourly["datetime"])

# Add hour
hourly["hour"] = hourly["datetime"].dt.hour

# Create figures folder
import os
os.makedirs("results/figures", exist_ok=True)

# 1. AC Power vs Irradiation
plt.figure(figsize=(8, 5))
plt.scatter(hourly["irradiation"], hourly["ac_power"])
plt.xlabel("Irradiation")
plt.ylabel("AC Power")
plt.title("AC Power vs Irradiation")
plt.grid()
plt.savefig(
    "results/figures/01_ac_power_vs_irradiation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 2. Module Temperature vs Ambient Temperature
plt.figure(figsize=(8, 5))
plt.scatter(
    hourly["ambient_temp"],
    hourly["module_temp"],
    c=hourly["irradiation"]
)
plt.xlabel("Ambient Temperature")
plt.ylabel("Module Temperature")
plt.title("Module Temperature vs Ambient Temperature")
plt.colorbar(label="Irradiation")
plt.grid()
plt.savefig(
    "results/figures/02_module_vs_ambient_temperature.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 3. AC Power vs DC Power
plt.figure(figsize=(8, 5))
plt.scatter(hourly["dc_power"], hourly["ac_power"])
plt.xlabel("DC Power")
plt.ylabel("AC Power")
plt.title("AC Power vs DC Power")
plt.grid()
plt.savefig(
    "results/figures/03_ac_vs_dc_power.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 4. Average AC Power by Hour
average_ac = hourly.groupby("hour")["ac_power"].mean()

plt.figure(figsize=(8, 5))
plt.plot(average_ac.index, average_ac.values, marker="o")
plt.xlabel("Hour of Day")
plt.ylabel("Average AC Power")
plt.title("Average AC Power by Hour")
plt.xticks(range(24))
plt.grid()
plt.savefig(
    "results/figures/04_average_ac_by_hour.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("EDA figures created successfully!")
