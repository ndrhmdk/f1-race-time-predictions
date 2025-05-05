import os
import fastf1 as ff1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error

ff1.Cache.enable_cache('f1-cache')

# Clear the sceen before printing the result
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Fetch, load, and prepare the data before modeling
def load_and_prepare_data():
    # Get [Race] Session
    session = ff1.get_session(2024, 3, "R")
    session.load()
    laps = session.laps[["Driver", "LapTime"]].copy()

    # Get [Qualifying] Session
    qualifying_session = ff1.get_session(2024, 3, "Q")
    qualifying_session.load()
    qualifying_laps = qualifying_session.laps.pick_quicklaps()
    qualifying_laps = qualifying_laps.groupby("Driver")["LapTime"].min().reset_index()
    qualifying_laps.rename(columns={"LapTime": "QualifyingTime"}, inplace=True)

    # Dropping all Null Values
    laps.dropna(subset=["LapTime"], inplace=True)
    qualifying_laps.dropna(subset=["QualifyingTime"], inplace=True)

    # Convert the LapTime and Qualifying Time into Second
    laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
    qualifying_laps["QualifyingTime (s)"] = qualifying_laps["QualifyingTime"].dt.total_seconds()

    # Mapping the Drivers' Names and their abbreviation
    driver_mapping = {
        "Alexander Albon": "ALB", "Fernando Alonso": "ALO", "Valtteri Bottas": "BOT", "Pierre Gasly": "GAS",
        "Lewis Hamilton": "HAM", "Nico Hülkenberg": "HUL", "Charles Leclerc": "LEC", "Kevin Magnussen": "MAG",
        "Lando Norris": "NOR", "Esteban Ocon": "OCO", "Sergio Pérez": "PER", "Oscar Piastri": "PIA",
        "Daniel Ricciardo": "RIC", "George Russell": "RUS", "Carlos Sainz": "SAI", "Lance Stroll": "STR",
        "Yuki Tsunoda": "TSU", "Max Verstappen": "VER", "Zhou Guanyu": "ZHO"
    }
    mapping = {v: k for k, v in driver_mapping.items()}

    qualifying = qualifying_laps[["Driver", "QualifyingTime (s)"]].copy()
    qualifying["Driver"] = qualifying["Driver"].map(mapping)
    qualifying["DriverCode"] = qualifying_laps["Driver"]
    qualifying = qualifying[["Driver", "DriverCode", "QualifyingTime (s)"]]

    # Finalizing Data for Model Constructing
    data = qualifying.merge(laps, left_on="DriverCode", right_on="Driver")
    data.drop(columns=["Driver_y", "LapTime"], inplace=True)
    data.rename(columns={"Driver_x": "Driver"}, inplace=True)

    return data

# Train and Evaluate Models
def train_and_evaluate_models(data):
    X = data[["QualifyingTime (s)"]]
    y = data["LapTime (s)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_metrics = {} # Dictionary to store model name and metrics

    def predict_and_metrics(model, model_name):
        df = data.copy()
        predicted = model.predict(df[["QualifyingTime (s)"]])
        df["PredictedRaceTime (s)"] = predicted

        df = df.drop_duplicates(subset=["Driver"]) # make sure the dataframe doesn't have any duplicates
        df = df.sort_values(by="PredictedRaceTime (s)") # to see who would be the winner

        y_pred = model.predict(X_test)
        mae = round(mean_absolute_error(y_test, y_pred), 4)
        rmse = round(root_mean_squared_error(y_test, y_pred), 4)
        r2 = round(r2_score(y_test, y_pred), 4)
        mape = round(mean_absolute_percentage_error(y_test, y_pred), 4)

        model_metrics[model_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
            "Predictions": df[["Driver", "PredictedRaceTime (s)"]].reset_index(drop=True)
        }

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42),
        "Stacking Regressor (rf, xgb, gb, ridgecv)": StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
            ],
            final_estimator=RidgeCV(), passthrough=True
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predict_and_metrics(model, name)

    return model_metrics

# Determine the Best Model
def find_best_model(model_metrics):
    result = pd.DataFrame(model_metrics).T.reset_index()
    result = result.rename(columns={'index': 'Model'})
    metrics_only = result.drop(columns=["Predictions"])

    df = metrics_only.copy()
    df = df.set_index("Model")

    lower_is_better = ["MAE", "RMSE", "MAPE"]
    higher_is_better = ["R2"]

    for metric in lower_is_better:
        df[f"{metric}_rank"] = df[metric].rank(ascending=True)
    for metric in higher_is_better:
        df[f"{metric}_rank"] = df[metric].rank(ascending=False)

    rank_cols = [col for col in df.columns if col.endswith("_rank")]
    df["Average_Rank"] = df[rank_cols].mean(axis=1)

    best_model_row = df.sort_values("Average_Rank").iloc[0]
    best_model_name = df.sort_values("Average_Rank").index[0]

    return best_model_name, result, model_metrics