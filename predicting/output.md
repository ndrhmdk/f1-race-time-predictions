The final output would be like this:
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
3    Charles Leclerc              83.681177
4       Lando Norris              83.693734
5      Oscar Piastri              83.987117
6    Fernando Alonso              84.144654
7     George Russell              84.160636
8       Lance Stroll              84.224564
9       Yuki Tsunoda              84.233696
10    Lewis Hamilton              84.430046
11   Alexander Albon              84.624113
12   Valtteri Bottas              84.863842
13   Kevin Magnussen              84.963159
14  Daniel Ricciardo              85.007680
15      Esteban Ocon              85.180057
16   Nico Hülkenberg              85.589880
17      Pierre Gasly              85.596730
18       Zhou Guanyu              85.831893

🏁 The predicted Winner is Max Verstappen with a predicted time of 83.237106 seconds.
```