#from sklearn.metrics import accuracy_score
from preprocess import load_data, make_preprocessor, split_train_test
from train_models import train_models
from test_performance import evaluate_models
from feature_importance import feature_importance
from tuning import tune_models
from roc_curve import plot_roc_curves

DATA_PATH = "../data/marketing_campaign.csv"
TARGET_COL = "Response"
result_roc = "../results/roc/roc_final.png"

def main() -> None:
    # 1) Load dataset
    df = load_data(DATA_PATH)

    # 2) Data preprocessing
    preprocessor, X, y = make_preprocessor(df, TARGET_COL)

    # 3) Split data into train and test sets: 20% for test set
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 4) Baseline training
    trained = train_models(preprocessor, X_train, y_train)

    # 5) Hyperparameter tuning
    tuned, cv_summary = tune_models(preprocessor, X_train, y_train)
    print("\nGridSearchCV summary (best CV accuracy):")
    print(cv_summary.to_string(index=False, float_format="{:.3f}".format))
    print("\n-----------------------------------------------------")

    # 6) Evaluate the models
    # feature_importance models
    results_df_baseline, cm_baseline = evaluate_models(trained, X_test, y_test)
    print("Performance summary for feature_importance models\n")
    print(results_df_baseline.to_string(col_space=12, float_format="{:.3f}".format))

    # tuned models
    results_df_tuned, cm_tuned = evaluate_models(tuned, X_test, y_test)
    print("Performance summary for tuned models\n")
    print(results_df_tuned.to_string(col_space=12, float_format="{:.3f}".format))

    # 7. Select final models
    FINAL_BASELINE = ["XGBoost", "AdaBoost"]
    trained_final = {k: v for k, v in trained.items() if k in FINAL_BASELINE}

    FINAL_TUNED = ["Random Forest", "Decision Tree"]
    tuned_final = {k: v for k, v in tuned.items() if k in FINAL_TUNED}

    # ----- print confusion matrices for the final models -----
    _, cm_final_baseline = evaluate_models(trained_final, X_test, y_test)
    print("\nConfusion matrices — FINAL feature_importance models")
    for name, cm in cm_final_baseline.items():
        print(f"\n{name}:\n{cm}")

    _, cm_final_tuned = evaluate_models(tuned_final, X_test, y_test)
    print("\nConfusion matrices — FINAL tuned models")
    for name, cm in cm_final_tuned.items():
        print(f"\n{name}:\n{cm}")

    # 8. Generate ROC curves for final models
    final_models = {**trained_final, **tuned_final}
    plot_roc_curves(final_models, X_test,y_test,out_path=result_roc,title="ROC Curves",)

    # 9. Print Feature-importance table
    feature_importance(trained_final, preprocessor, top_n=5)
    feature_importance(tuned_final, preprocessor, top_n=5)

if __name__ == "__main__":
    main()
