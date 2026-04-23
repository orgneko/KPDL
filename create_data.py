from ucimlrepo import fetch_ucirepo
import pandas as pd

# 1. Tải dữ liệu từ UCI
student_performance = fetch_ucirepo(id=320)
X = student_performance.data.features
y = student_performance.data.targets

# 2. Kết hợp lại thành một bảng duy nhất
df = pd.concat([X, y], axis=1)

# 3. Lưu thành file CSV vật lý (dùng dấu chấm phẩy theo đúng chuẩn bộ dữ liệu này)
df.to_csv('student-mat.csv', sep=';', index=False)

print("✅ Đã tạo file student-mat.csv thành công!")
