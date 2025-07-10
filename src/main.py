from pathlib import Path
from preprocess import load_data, make_preprocessor, split_train_test
from train_models import train_and_compare

DATA_PATH = "../data/marketing_campaign.csv"
TARGET_COL = "Response"


def main() -> None:
    # 1) Load dataset
    df = load_data(DATA_PATH)

    # 2) Build preprocessing pipeline and split X / y
    preprocessor, X, y = make_preprocessor(df, TARGET_COL)

    # 3) Train-test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("Train shape:", X_train.shape, y_train.shape) # Train shape: (1792, 28) (1792,)
    print("Test  shape:", X_test.shape, y_test.shape) # Test  shape: (448, 28) (448,)

    # 4) Train and compare multiple models
    results_df, _ = train_and_compare(preprocessor, X_train, y_train, X_test, y_test)

    print("\nModel performance (accuracy on test set):")
    print(results_df.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
