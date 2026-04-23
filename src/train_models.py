from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt
import os


def train_and_evaluate(X_train, X_test, y_train, y_test):
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    }
    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        trained_models[name] = model
        print(f"{name} Accuracy: {acc*100:.2f}%")

    return results, trained_models


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rớt (0)', 'Đậu (1)'],
                yticklabels=['Rớt (0)', 'Đậu (1)'])
    plt.title(f'Ma trận nhầm lẫn - {model_name}')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')

    # Lưu kết quả
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(f'results/confusion_matrix_{model_name}.png')
    plt.show()
# Ý hiểu: Bước này giúp ta biết thuật toán nào 'thông minh' nhất với dữ liệu sinh viên.
