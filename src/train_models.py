from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        results[name] = acc
        print(f"--- {name} ---")
        print(f"Độ chính xác: {acc*100:.2f}%")
        print(classification_report(y_test, predictions))

    return results, models

# Ý hiểu: Bước này giúp ta biết thuật toán nào 'thông minh' nhất với dữ liệu sinh viên.
