from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
import os


def train_and_evaluate(X, y):
    print("Bắt đầu training model...")

    # 1. CHỌN FEATURE QUAN TRỌNG (GIẢM INPUT)
    selected_features = [
        'G1', 'G2',
        'failures',
        'absences',
        'studytime',
        'goout'
    ]

    X = X[selected_features]

    print("Các feature được sử dụng:", selected_features)

    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Danh sách model
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
    }

    results = {}
    trained_models = {}

    # 4. Train từng model (Pipeline chỉ còn SMOTE + Model)
    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipeline = ImbPipeline(steps=[
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results[name] = acc
        trained_models[name] = pipeline

        print(f"{name} Accuracy: {acc * 100:.2f}%")

    # 5. In report cho model tốt nhất (Random Forest)
    print("\nClassification Report (Random Forest):")
    print(classification_report(
        y_test, trained_models["Random Forest"].predict(X_test)))

    # RETURN ĐÚNG CHO MAIN
    return results, trained_models, X_test, y_test


#  VẼ CONFUSION MATRIX
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rớt (0)', 'Đậu (1)'],
                yticklabels=['Rớt (0)', 'Đậu (1)'])

    plt.title(f'Ma trận nhầm lẫn - {model_name}')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')

    if not os.path.exists('results'):
        os.makedirs('results')

    plt.savefig(f'results/confusion_matrix_{model_name}.png')
    plt.show()
