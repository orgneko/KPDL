import pandas as pd
import os


def load_data(file_name='student-mat.csv'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {file_path}")

    df = pd.read_csv(file_path, sep=';')

    # Target: Đậu / Rớt
    y = (df['G3'] >= 10).astype(int)

    # Feature chọn lọc
    features = [
        'G1', 'G2',
        'failures',
        'absences',
        'studytime',
        'goout',
        'schoolsup', 'famsup', 'paid',
        'activities', 'nursery',
        'higher', 'internet', 'romantic'
    ]

    # Check thiếu cột
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột: {missing_cols}")

    X = df[features].copy()

    # Encode yes/no
    binary_cols = [
        'schoolsup', 'famsup', 'paid',
        'activities', 'nursery',
        'higher', 'internet', 'romantic'
    ]

    for col in binary_cols:
        X[col] = X[col].map({'yes': 1, 'no': 0}).fillna(0)

    print(f"Đã tải dữ liệu thành công từ: {file_path}")
    print(f"Số mẫu: {len(X)} | Số feature: {len(X.columns)}")

    return X, y


if __name__ == "__main__":
    X, y = load_data()
    print(X.head())
