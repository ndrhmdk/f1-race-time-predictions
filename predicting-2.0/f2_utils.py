import os
import fastf1 as ff1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error

ff1.Cache.enable_cache('f1-cache')

# Clear the screen before printing the result
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Fetch, load, and prepare the data before modeling
def load_and_prepare_data():
    # Get [Race] session
    session = ff1.get_session(2025, 3, "R")
    session.load()
    laps = session.laps[[
        "Driver", "LapTime",
        "Sector1Time", "Sector2Time", "Sector3Time"
    ]]
    laps.dropna(inplace=True)

    # Get [Qualifying] session
    qualifying_session = ff1.get_session(2025, 3, "Q")
    qualifying_session.load()
    qualifying_laps = qualifying_session.laps.pick_quicklaps()
    qualifying_laps = qualifying_laps.groupby("Driver")["LapTime"].min().reset_index()
    qualifying_laps.rename(columns={"LapTime": "QualifyingTime"}, inplace=True)

    # Convert the LapTime and QualifyingTime to second
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()
    qualifying_laps["QualifyingTime (s)"] = qualifying_laps["QualifyingTime"].dt.total_seconds()

    # Group by driver to get everage sector times per driver
    sector_times = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

    driver_mapping = {
        "Alexander Albon": "ALB", "Fernando Alonso": "ALO", "Andrea Kimi Antonelli": "ANT", "Oliver Bearman": "BEA", 
        "Gabriel Bortoleto": "BOR", "Jack Doohan": "DOO", "Pierre Gasly": "GAS", "Isack Hadjar": "HAD",
        "Lewis Hamilton": "HAM", "Nico Hülkenberg": "HUL", "Liam Lawson": "LAW", "Charles Leclerc": "LEC",
        "Lando Norris": "NOR", "Esteban Ocon": "OCO", "Oscar Piastri": "PIA", "George Russell": "RUS",
        "Carlos Sainz Jr.": "SAI", "Lance Stroll": "STR", "Yuki Tsunoda": "TSU", "Max Verstappen": "VER"
    }
    mapping = {v: k for k, v in driver_mapping.items()}

    qualifying = qualifying_laps[["Driver", "QualifyingTime (s)"]].copy()
    qualifying["Driver"] = qualifying["Driver"].map(mapping)
    qualifying["DriverCode"] = qualifying_laps["Driver"]
    qualifying = qualifying[["Driver", "DriverCode", "QualifyingTime (s)"]]
    
    # Finalizing Data for Model Constructing
    data = qualifying.merge(sector_times, left_on="DriverCode", right_on="Driver", how="left")
    data.drop(columns=["Driver_y"], inplace=True)
    data.rename(columns={"Driver_x": "Driver"}, inplace=True)

    return laps, data, qualifying

# Train and Evaluate Model
def train_and_evaluate_model(laps, data, qualifying):
    X = data.drop(columns=["Driver", "DriverCode"])
    y = laps.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    predicted_race_times = model.predict(X)
    qualifying["PredictedRaceTime (s)"] = predicted_race_times
    qualifying = qualifying.sort_values(by="PredictedRaceTime (s)")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return qualifying, mae, r2, mape, rmse