from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import sys

# Đảm bảo tìm thấy module trong src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# Tải mô hình
MODEL_PATH = os.path.join('..', 'models', 'best_student_model.pkl')
model = None
features_in = []

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    features_in = model.feature_names_in_
    print("✅ Đã tải mô hình thành công!")
else:
    print("❌ LỖI: Không tìm thấy file .pkl. Hãy chạy main.py trước!")


@app.route('/')
def index():
    # Khi mới vào trang chủ, không hiện kết quả (prediction=None)
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Khởi tạo dữ liệu mặc định ở mức "trung bình" (số 2)
        # Điều này giúp các yếu tố không nhập vào không kéo kết quả xuống quá tệ
        input_values = {col: 2 for col in features_in}

        # 2. Lấy dữ liệu từ Web form
        raw_g1 = float(request.form.get('G1', 0))
        raw_g2 = float(request.form.get('G2', 0))
        raw_failures = int(request.form.get('failures', 0))
        raw_absences = int(request.form.get('absences', 0))
        raw_studytime = int(request.form.get('studytime', 1))
        raw_goout = int(request.form.get('goout', 1))

        # 3. Đưa dữ liệu vào dictionary
        for col in input_values.keys():
            col_lower = col.lower()
            if col_lower == 'g1':
                input_values[col] = raw_g1 * 2
            elif col_lower == 'g2':
                input_values[col] = raw_g2 * 2
            elif col_lower == 'failures':
                input_values[col] = raw_failures
            elif col_lower == 'absences':
                input_values[col] = raw_absences
            elif col_lower == 'studytime':
                input_values[col] = raw_studytime
            elif col_lower == 'goout':
                input_values[col] = raw_goout

        # 4. Tạo DataFrame đúng cấu trúc
        input_df = pd.DataFrame([input_values], columns=features_in)

        # 5. DỰ ĐOÁN THUẦN TÚY TỪ AI (KHÔNG CÓ LUẬT ÉP BUỘC)
        probability = model.predict_proba(input_df)[0]
        prob_pass = probability[1]  # Xác suất ĐẬU (từ 0.0 đến 1.0)

        # Quyết định Đậu/Rớt dựa trên ngưỡng 0.5 (50%)
        if prob_pass >= 0.5:
            result_text = "ĐẬU (PASS)"
            status = "safe"
        else:
            result_text = "CÓ NGUY CƠ TRƯỢT (FAIL)"
            status = "fraud"

        # Con số này sẽ nhảy tự do: 35.5%, 52.1%, 12.0%... tùy vào nỗ lực của sinh viên
        return render_template('index.html',
                               prediction=result_text,
                               confidence=round(prob_pass * 100, 2),
                               status=status)

    except Exception as e:
        return f"Đã có lỗi xảy ra: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
