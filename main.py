from src.data_preprocessing import load_and_clean_data
from src.train_models import train_and_evaluate
from src.visualization import *


def main():
    print("🚀 Khởi động hệ thống dự đoán hành vi học tập...")

    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_clean_data()

    # 2. Train
    results, trained_models = train_and_evaluate(
        X_train, X_test, y_train, y_test)

    plot_model_comparison(results)

    # 3. Visualize với mô hình tốt nhất (thường là Random Forest)
    best_model = trained_models["Random Forest"]
    plot_feature_importance(best_model, X_train.columns)


if __name__ == "__main__":
    main()
