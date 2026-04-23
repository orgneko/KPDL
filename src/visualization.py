import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_feature_importance(model, feature_names):

    if not os.path.exists('results'):
        os.makedirs('results')

    importances = model.feature_importances_
    indices = importances.argsort()[-10:]  # Lấy 10 yếu tố quan trọng nhất

    plt.figure(figsize=(10, 6))
    plt.title('Top 10 yếu tố ảnh hưởng đến kết quả học tập')
    plt.barh(range(len(indices)),
             importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Mức độ ảnh hưởng')

    plt.savefig('results/feature_importance.png')
    print("Đã lưu biểu đồ vào results/feature_importance.png")

    plt.show()


def plot_model_comparison(results_dict):
    # Vẽ biểu đồ so sánh độ chính xác của 3 thuật toán
    plt.figure(figsize=(8, 5))
    names = list(results_dict.keys())
    values = list(results_dict.values())

    sns.barplot(x=names, y=values, palette='viridis')
    plt.title('So sánh độ chính xác giữa các mô hình')
    plt.ylim(0, 1.0)

    # Lưu ảnh
    plt.savefig('results/model_comparison.png')
    print("Đã lưu biểu đồ so sánh vào results/model_comparison.png")
    plt.show()
