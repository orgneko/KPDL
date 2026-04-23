import pandas as pd
import joblib


def predict_new_student():
    try:
        model = joblib.load('models/best_student_model.pkl')
        feature_names = model.feature_names_in_
        expected_count = len(feature_names)  # Lấy đúng số lượng máy cần (30)
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    # 2. Dữ liệu một sinh viên mới
    # Mình sẽ tạo một danh sách có 32 giá trị như bạn làm
    raw_data = [
        0, 1, 18, 1, 0, 1, 4, 4, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 4, 3, 4, 1, 2, 5, 15, 7, 5
    ]

    # CHỈ LẤY ĐÚNG SỐ LƯỢNG MÀ MÁY CẦN (Cắt bớt từ đầu cho đến vị trí thứ 30)
    final_data = [raw_data[:expected_count]]

    df_new = pd.DataFrame(final_data, columns=feature_names)

    # 3. Dự đoán
    prediction = model.predict(df_new)
    prob = model.predict_proba(df_new)

    print("\n" + "="*30)
    print(f"KẾT QUẢ PHÂN TÍCH (Dựa trên {expected_count} đặc trưng)")
    if prediction[0] == 1:
        print(f"Dự đoán: ĐẬU (Tỷ lệ: {prob[0][1]*100:.2f}%)")
    else:
        print(
            f"Dự đoán: NGUY CƠ TRƯỢT (Tỷ lệ: {prob[0][0]*100:.2f}%)")
    print("="*30)


if __name__ == "__main__":
    predict_new_student()
