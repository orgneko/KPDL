import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_clean_data():
    # 1. Tải dữ liệu
    dataset = fetch_ucirepo(id=320)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # 2. Xử lý tất cả các cột chữ (Categorical Data) thành số
    le = LabelEncoder()
    for col in X.columns:
        # Nếu cột là kiểu chữ (object) hoặc không phải số
        if X[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = le.fit_transform(X[col].astype(str))

    # 3. Chuyển mục tiêu G3 thành Nhị phân (Đậu/Rớt)
    # Lưu ý: Ép kiểu y['G3'] về số trước khi so sánh để tránh lỗi 'GP'
    y_numeric = pd.to_numeric(y['G3'], errors='coerce').fillna(0)
    y_binary = y_numeric.apply(lambda x: 1 if x >= 10 else 0)

    # 4. Chia dữ liệu
    return train_test_split(X, y_binary, test_size=0.2, random_state=42)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_clean_data()
    print("✅ Tiền xử lý hoàn tất. Dữ liệu đã sẵn sàng!")
