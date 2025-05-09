# 🏎️ **F1 Race Time Prediction – "*Let's ride, let's ride, let's ride, let's ride...*"**

**Insprised by** Charli xcx - *Vroom Vroom*

![alt text](<project github.png>)

## 🚀 **Project Overview**
After vibing to Charli xcx's *Vroom Vroom* and watched some videos of [Mar Antaya](https://github.com/mar-antaya) building her Machine Learning model to predict the result of the F1 race, I decided to try it too! 

This project predicts F1 race times based on qualifying times, using real-world telemetry data and sector performance from actual F1 sessions.

“Let’s ride, let’s ride, let’s ride, let’s ride” — Charli XCX
That’s the energy that fueled this engine. 🏁

### 🎯 **Project Goal**
Using qualifying data (and optionally sector breakdowns) to predict the average **race lap time** for each driver.

Two main versions:
* **Basic**: Uses just qualifying times.
* **Advanced**: Addes sector times and averages per driver for better granularity.

### 🧱 **Tech Stack**
* **Language**: Python
* **Data Source**: from [`FastF1`](https://pypi.org/project/fastf1/) API.
* **Libraries used**: `fastf1`, `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `sklearn`, `xgboost`...
* **Modeling Methods**: Linear Regression, RidgeCV, Random Forest Regressor, Gradient Boosting Regressor, Stacking Regressor, XGB Regressor...
* **Metrics**: MAE, RMSE, R² Score, MAPE

## ⚙️ **Project Workflow**
### **1. 🏁 Data Collection**:
* Pulled real telemetry and timing data from F1 sessions using `FastF1`.
* Cleaned and transformed lap and qualifying times into numerical seconds.
* Mapped driver codes (e.g., `VER` → Max Verstappen).
### **2. 🧪 Feature Engineering**
* **Basic version**: Uses jsut the driver's fastest qualifying lap time.
* **Advanced version**: Adds mean sector times per driver.

### **3. 🤖 Model Training & Evaluation**
* Models trained on Qualifying Time → Race Lap Time
* Evaluated using:
    * Mean Absolute Error (MAE)
    * Root Mean Squared Error (RMSE)
    * Mean Absolute Percentage Error (MAPE)
    * R² Score

### **4. 🏆 Model Selection**
* All models ranked on average rank across metrics.
* Best model selected automatically.
* Final output: sorted prediction list + winner announcement.

### **📈 Sample Output**
**Basic Version**
```text
--- Model Metrics Comparison ---
                                       Model     MAE    RMSE      R2    MAPE
0                          Linear Regression  3.7989  6.9325 -0.0009   0.041
1                    Random Forest Regressor  3.9178  7.1142  -0.054  0.0423
2                Gradient Boosting Regressor   3.957  7.1249 -0.0572  0.0428
3                          XGBoost Regressor  3.9281  7.1035 -0.0509  0.0425
4  Stacking Regressor (rf, xgb, gb, ridgecv)  3.8137  6.8776  0.0149  0.0413

🏆 Best Overall Model: Linear Regression

📋 Predicted Race Times:
              Driver  PredictedRaceTime (s)
0     Max Verstappen              83.237106
1       Carlos Sainz              83.545330
2       Sergio Pérez              83.646930
...
16   Nico Hülkenberg              85.589880
17      Pierre Gasly              85.596730
18       Zhou Guanyu              85.831893

🏁 The predicted Winner is Max Verstappen with a predicted time of 83.237106 seconds.
```

**Advanced Version**
```text
🏁 Predicted 2025 Chinese GP Winner with New Drivers and Sector Times 🏁

                   Driver  PredictedRaceTime (s)
19         Max Verstappen              92.928038
14          Oscar Piastri              92.935192
12           Lando Norris              92.935481
...
9         Nico Hülkenberg              94.328500
13           Esteban Ocon              94.339577
10            Liam Lawson              94.351038

🔍 Model Evaluation:
- MAE: 0.2032
- R2 Score: 0.6134
- MAPE: 0.0021
- RMSE: 0.362405
```

## **🗂️ File Structure**
```bash
📁 f1-race-predictor/
├── 📁 predicting/              # Basic version
│   ├── __pycache__/
│   ├── f1_utils.py                 
│   ├── main.py                     
│   └── output.md                   
├── 📁 predicting-2.0/          # Advanced version
│   ├── f2_utils.py
│   ├── main.py
│   └── output.md
├── README.md
├── f1-prediction.ipynb         # Basic verion (notebook)
├── f1-predictions-2.ipynb      # Advanced version (notebook)
└── project_github.png          # Visual asset for README
```

## **🧩 Future Improvements**
* Include the affection of the weather to the race.
* Predict total race time, not just average lap.
* Use telemetry data (braking, throttle, speed traps) as features.
* Try deep learning models (e.g., neural nets).
* Extend to predicting podium finishes or pit stop timing

## **🎵 Vibe Check**
> “Let’s ride...”<br>— Charli XCX, but also your regression model probably...

## **📬 Contact**
Since I am quite new to Machine Learning, Deep Learning ... and I'm still working on my skills, so it would be nice if you guys can give me some tips and method that could help me improve my skills! 

Feel free to reach out via [LinkedIn](https://www.linkedin.com/in/hmdkien/) or [Gmail](andrhmdk@gmail.com). Thank you!
