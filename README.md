# 🎓 Student Success Predictor (KPDL Project)

Hệ thống dự đoán kết quả học tập của sinh viên dựa trên kỹ thuật Khai phá dữ liệu (Data Mining). Ứng dụng sử dụng mô hình Machine Learning để phân tích các yếu tố học thuật và hành vi nhằm đưa ra cảnh báo sớm về nguy cơ trượt môn.

## 📁 Cấu trúc thư mục

- `src/`: Chứa mã nguồn xử lý dữ liệu và huấn luyện mô hình.
- `web/`: Giao diện ứng dụng Web (Flask).
- `models/`: Lưu trữ mô hình đã huấn luyện (.pkl).
- `results/`: Biểu đồ đánh giá độ chính xác và ma trận nhầm lẫn.
- `main.py`: File thực thi chính để chạy quy trình từ đầu đến cuối.

## 🚀 Tính năng nổi bật

- **Mô hình đa dạng**: So sánh giữa Decision Tree, Naive Bayes và Random Forest.
- **Xử lý dữ liệu lệch**: Áp dụng kỹ thuật **SMOTE** và **Class Weight** để cải thiện khả năng dự báo nhóm sinh viên yếu.
- **Giao diện trực quan**: Dự báo xác suất đậu/rớt theo thời gian thực với Flask.

## 🛠 Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt môi trường

Đảm bảo bạn đã cài đặt Python 3.x. Sau đó cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình

Chạy file main.py để thực hiện lấy dữ liệu từ UCI, tiền xử lý và huấn luyện mô hình:

```bash
python main.py
```

Sau khi chạy, file best_student_model.pkl sẽ xuất hiện trong folder models/.

### 3. Khởi chạy ứng dụng Web

Di chuyển vào thư mục web và khởi động server:

```bash
cd web
python app.py
```

Truy cập ứng dụng tại địa chỉ: http://127.0.0.1:5000
Kết quả mô hình
Dự án tập trung vào việc tối ưu hóa khả năng nhận diện các sinh viên có nguy cơ (Rớt), chấp nhận đánh đổi một phần độ chính xác tổng thể (Accuracy) để tăng cường độ nhạy thực tế cho hệ thống cảnh báo.
Thực hiện bởi: [orgneko]
