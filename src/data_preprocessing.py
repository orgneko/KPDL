import pandas as pd
from ucimlrepo import fetch_ucirepo


import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_data():
    dataset = fetch_ucirepo(id=320)

    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # 🎯 Tạo target từ G3
    y = pd.to_numeric(df['G3'], errors='coerce').fillna(0)
    y = (y >= 10).astype(int)

    # 🎯 X = toàn bộ trừ G3
    X = df.drop(columns=['G3'])

    print("✅ Columns hiện tại:", X.columns)

    return X, y


if __name__ == "__main__":
    X, y = load_data()
    print(X.head())
