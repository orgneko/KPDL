from src.data_preprocessing import load_and_clean_data
from src.train_models import train_and_evaluate, plot_confusion_matrix
from src.visualization import *

import joblib
import os

if not os.path.exists('models'):
    os.makedirs('models')


def main():
    print(" Khởi động hệ thống dự đoán hành vi học tập...")

    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_clean_data()

    # 2. Train
    results, trained_models = train_and_evaluate(
        X_train, X_test, y_train, y_test)
    # 3. Vẽ biểu dồ
    plot_model_comparison(results)
    # 4. Visualize với mô hình tốt nhất (thường là Random Forest)
    best_name = max(results, key=results.get)
    best_model = trained_models["Random Forest"]

    print(f"Thuật toán tốt nhất là: {best_name}")
    # 5. vẽ ma trận nhầm lẫn
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, best_name)
    # 6. vẽ biểu đồ các yếu tố quan trọng
    plot_feature_importance(best_model, X_train.columns)

    model_path = f'models/best_student_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"Đã đóng gói bộ não ({best_name}) vào: {model_path}")


if __name__ == "__main__":
    main()
