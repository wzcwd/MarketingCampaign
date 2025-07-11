#from sklearn.metrics import accuracy_score
from preprocess import load_data, make_preprocessor, split_train_test
from train_models import train_models
from test_performance import evaluate_models



DATA_PATH = "../data/marketing_campaign.csv"
TARGET_COL = "Response"

def main() -> None:
    # 1) Load dataset
    df = load_data(DATA_PATH)

    # 2) Build preprocessing pipeline and split X / y
    preprocessor, X, y = make_preprocessor(df, TARGET_COL)

    # 3) Train-test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # print("Train shape:", X_train.shape, y_train.shape) # Train shape: (1792, 28) (1792,)
    # print("Test  shape:", X_test.shape, y_test.shape) # Test  shape: (448, 28) (448,)

    # 4) Train models
    trained = train_models(preprocessor, X_train, y_train)
    # for model_name, estimator in trained.items():
    #     y_pred = estimator.predict(X_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     print(f"{model_name:15s}  Accuracy: {acc:.4f}")

    # 5) Evaluate on test set
    results_df, _ = evaluate_models(trained, X_test, y_test)

    print("Performance summary for test set\n")
    print(results_df.to_string(col_space=12, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
