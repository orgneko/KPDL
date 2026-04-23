from src.data_preprocessing import load_data
from src.train_models import train_and_evaluate, plot_confusion_matrix
from src.visualization import *

import joblib
import os


# Tạo thư mục models nếu chưa có
if not os.path.exists('models'):
    os.makedirs('models')


def main():
    print("Khởi động hệ thống dự đoán...")

    # 1. Load data
    X, y = load_data()

    # 2. Train + evaluate
    results, trained_models, X_test, y_test = train_and_evaluate(X, y)

    # 3. Vẽ biểu đồ so sánh model
    plot_model_comparison(results)

    # 4. Lấy model tốt nhất
    best_model = trained_models["Random Forest"]
    best_name = "Random Forest"

    print(f"\nSử dụng model để visualize: {best_name}")

    # 5. Ma trận nhầm lẫn
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, best_name)

    # 6. Feature Importance (FIX CHUẨN)
    try:
        model_step = best_model.named_steps['model']  # lấy RandomForest

        feature_names = [
            'G1', 'G2',
            'failures',
            'absences',
            'studytime',
            'goout'
        ]

        plot_feature_importance(model_step, feature_names)

    except Exception as e:
        print("⚠️ Không thể vẽ feature importance:", e)

    # 7. Lưu model
    model_path = 'models/best_student_model.pkl'
    joblib.dump(best_model, model_path)

    print(f"\nĐã lưu model vào: {model_path}")
    print("👉 Model đã bao gồm xử lý → dùng trực tiếp cho Flask")


if __name__ == "__main__":
    main()
