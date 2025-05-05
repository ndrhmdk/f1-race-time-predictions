from f1_utils import load_and_prepare_data, train_and_evaluate_models, find_best_model, clear_screen

def main():
    data = load_and_prepare_data()
    model_metrics = train_and_evaluate_models(data)
    best_model_name, result, model_metrics = find_best_model(model_metrics)

    clear_screen()

    print("\n--- Model Metrics Comparison ---")
    print(result.drop(columns=["Predictions"]))

    print(f"\n🏆 Best Overall Model: {best_model_name}")
    print("\n📋 Predicted Race Times:")
    print(model_metrics[best_model_name]["Predictions"])

    winner = model_metrics[best_model_name]["Predictions"].iloc[0]
    print(f"\n🏁 The predicted Winner is {winner['Driver']} with a predicted time of {winner['PredictedRaceTime (s)']:.6f} seconds.")

if __name__ == '__main__':
    main()