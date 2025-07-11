#from sklearn.metrics import accuracy_score
from preprocess import load_data, make_preprocessor, split_train_test
from train_models import train_models
from test_performance import evaluate_models
from feature_importance import feature_importance
from tuning import tune_models



DATA_PATH = "../data/marketing_campaign.csv"
TARGET_COL = "Response"
result_dir_baseline = "../results/baseline"
result_dir_tuned = "../results/tuned"

def main() -> None:
    # 1) Load dataset
    df = load_data(DATA_PATH)

    # 2) Build preprocessing pipeline and split X / y
    preprocessor, X, y = make_preprocessor(df, TARGET_COL)

    # 3) Train-test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # print("Train shape:", X_train.shape, y_train.shape) # Train shape: (1792, 28) (1792,)
    # print("Test  shape:", X_test.shape, y_test.shape) # Test  shape: (448, 28) (448,)

    # 4) Baseline training
    trained = train_models(preprocessor, X_train, y_train)
    # for model_name, estimator in trained.items():
    #     y_pred = estimator.predict(X_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     print(f"{model_name:15s}  Accuracy: {acc:.4f}")

    # 5) Hyper-parameter tuning
    tuned, cv_summary = tune_models(preprocessor, X_train, y_train)
    print("\nGridSearchCV summary (best CV accuracy):")
    print(cv_summary.to_string(index=False, float_format="{:.3f}".format))
    
    # 6) Evaluate the models
    # baseline models
    results_df_baseline, _ = evaluate_models(trained, X_test, y_test)
    print("Performance summary for baseline models\n")
    print(results_df_baseline.to_string(col_space=12, float_format="{:.3f}".format))

    # tuned models
    results_df_tuned, _ = evaluate_models(tuned, X_test, y_test)
    print("Performance summary for tuned models\n")
    print(results_df_tuned.to_string(col_space=12, float_format="{:.3f}".format))

    # 7) Plot feature importance (for models that support it)
    # Baseline
    feature_importance(trained, preprocessor, result_dir_baseline)
    # Tuned
    feature_importance(tuned, preprocessor, result_dir_tuned)
    print("\nFeature-importance plots saved")

if __name__ == "__main__":
    main()
