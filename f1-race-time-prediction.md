# 🏎️ **F1 Race Time Prediction** – *"Let's ride, let's ride, let's ride, let's ride..."*  
**Inspired by Charli XCX - *Vroom Vroom***

## 🚀 **Project Overview**
After vibing to Charli XCX's *"Vroom Vroom"*, I got the idea to combine two passions—Formula 1 and Machine Learning. This project aims to **predict F1 Race Time** based on **Qualifying Time** using real-world data from 19 different F1 drivers.  

> *"Let's ride, let's ride, let's ride, let's ride"* — Charli XCX 

> That's the energy that drove this project forward. 🏁

---

## 🧠 **Goal**
Given the **Qualifying Time** for each driver, predict their **Race Time** using various regression models. Each driver is represented by their standard F1 abbreviation.

## 📦 **Project Workflow**
This project consists of four main stages.
### 1. 🏁 **Data Mining**
To collect the data, I used the [`FastF1`](https://theoehrly.github.io/Fast-F1/) library, a Python API that allows access to real-time and historical Formula 1 timing data, including qualifying and race sessions. I fetched:
- **Qualifying time** for each driver
    ```python
    quali_ses = fastf1.get_session(2024, 3, "Q")
    ```
- **Race time** (total session time)
    ```python
    session = fastf1.get_session(2024, 3, "R") 
    ```

Other libraries used in this step:
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib`, `seaborn` - for visualizations
- `scikit-learn`, `xgboost` - for machine learning modeling

### 2. 🧹 **Data Preprocessing**
Before feeding the data into models, I performed several preprocessing tasks:
- Merged qualifying and race data by driver and Grand Prix
- Since there was no missing value, there was no need to do a missing value process.
- Converted lap and session times to consisten formats (in seconds)
- Create a data with the following columns: `Driver`, `Qualifying Time (s)`, `DriverCode`, `LapTime (s)`.

Here are all the drivers in the data
| **Abbreviation** | **Driver** Name         |
|--------------|---------------------|
| ALB          | Alexander Albon     |
| ALO          | Fernando Alonso     |
| BOT          | Valtteri Bottas     |
| GAS          | Pierre Gasly        |
| HAM          | Lewis Hamilton      |
| HUL          | Nico Hülkenberg     |
| LEC          | Charles Leclerc     |
| MAG          | Kevin Magnussen     |
| NOR          | Lando Norris        |
| OCO          | Esteban Ocon        |
| PER          | Sergio Pérez        |
| PIA          | Oscar Piastri       |
| RIC          | Daniel Ricciardo    |
| RUS          | George Russell      |
| SAI          | Carlos Sainz        |
| STR          | Lance Stroll        |
| TSU          | Yuki Tsunoda        |
| VER          | Max Verstappen      |
| ZHO          | Zhou Guanyu         |

### 3. 🤖 **Modeling**
The target variable is **Race Time**, and the main feature used is **Qualifying Time**. Multiple regression models were trained and compared:

#### ✅ **Models Used**:
**Linear Regression** – A classic, simple baseline model that assumes a linear relationship between Qualifying and Race Time.
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

**Random Forest Regressor** – A powerful ensemble method that uses multiple decision trees to reduce variance and improve predictions.
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42
)
```

- **Gradient Boosting Regressor** – Boosts weak learners iteratively, optimizing performance by minimizing prediction errors.
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=200, 
    max_depth=4, 
    learning_rate=0.1, 
    random_state=42
)
```

**XGBoost Regressor** – Highly optimized gradient boosting library
```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)
```

**Stacking Regressor** – Combines multiple base regressors (Random Forest, XGBoost, Gradient Boosting) and uses a meta-model (RidgeCV) to learn from their outputs. This can often outperform individual models.
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ],
    final_estimator=RidgeCV(),
    passthrough=True
)
```

### 4. 📈 **Evaluation**
Models were evaluated using **MSE (Mean Absolute Error)**, the formula of **MSE**:
$$\text{MSE}=\frac{1}{n}\sum^n_{i=1}|y_i-x_i|$$
* $\text{MSE}$: mean absolute erro
* $y_i$: prediction
* $x_i$: true value
* $n$: total number of data points

# **Next Step**
* Improve feature engineering
* Try deep learning models (e.g., neural nets)
* Extend to predicting podium finishes or pit stop timing

> Since machine learning is quite new to me, so there might be some errors, something that might not be too effective... So please reach out to me to let me know how I could improve this via my [`Gmail`](andrhmdk@gmail.com)!