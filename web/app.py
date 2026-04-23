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
    print("Đã tải mô hình thành công!")
else:
    print("LỖI: Không tìm thấy file .pkl. Hãy chạy main.py trước!")


@app.route('/')
def index():
    # Khi mới vào trang chủ, không hiện kết quả (prediction=None)
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'G1': float(request.form.get('G1')) * 2,
            'G2': float(request.form.get('G2')) * 2,
            'failures': int(request.form.get('failures')),
            'absences': int(request.form.get('absences')),
            'studytime': int(request.form.get('studytime')),
            'goout': int(request.form.get('goout'))
        }

        input_df = pd.DataFrame([data])

        probability = model.predict_proba(input_df)[0]
        prob_pass = probability[1]

        if prob_pass >= 0.5:
            result_text = "ĐẬU (PASS)"
            status = "safe"
            confidence = prob_pass * 100
        else:
            result_text = "CÓ NGUY CƠ TRƯỢT (FAIL)"
            status = "fraud"
            confidence = (1 - prob_pass) * 100
        return render_template(
            'index.html',
            prediction=result_text,
            status=status,
            confidence=round(prob_pass * 100, 2)
        )

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
