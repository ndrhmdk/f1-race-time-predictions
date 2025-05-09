
from f2_utils import load_and_prepare_data, clear_screen, train_and_evaluate_model

def main():
    laps, data, qualifying = load_and_prepare_data()
    qualifying, mae, r2, mape, rmse = train_and_evaluate_model(laps, data, qualifying)

    clear_screen()

    print("\n🏁 Predicted 2025 Chinese GP Winner with New Drivers and Sector Times 🏁\n")
    print(qualifying[["Driver", "PredictedRaceTime (s)"]])

    print(f"\n🔍 Model Evaluation:")
    print(f"- MAE: {mae:.4f}")
    print(f"- R2 Score: {r2:.4f}")
    print(f"- MAPE: {mape:.4f}")
    print(f"- RMSE: {rmse:4f}")

if __name__ == '__main__':
    main()