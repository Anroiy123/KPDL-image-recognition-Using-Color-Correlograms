"""
evaluate.py - Danh gia mo hinh va truc quan hoa ket qua

Tao cac bieu do:
- Confusion Matrix
- So sanh accuracy giua cac phuong phap
- Classification Report chi tiet
"""

import os
import sys
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Ve Confusion Matrix dang heatmap.

    Args:
        y_true: Nhan that
        y_pred: Nhan du doan
        class_names: Ten cac lop
        title: Tieu de bieu do
        save_path: Duong dan luu file anh
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    ax.set_xlabel('Du doan (Predicted)', fontsize=12)
    ax.set_ylabel('Thuc te (Actual)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Da luu: {save_path}")


def plot_accuracy_comparison(results, save_path):
    """Ve bieu do so sanh accuracy giua cac phuong phap.

    Args:
        results: List cac dict ket qua
        save_path: Duong dan luu file anh
    """
    labels = []
    accuracies = []
    colors = []

    color_map = {
        ('Correlogram', 'HSV', 'SVM'): '#2196F3',      # Xanh duong
        ('Correlogram', 'HSV', 'KNN'): '#4CAF50',      # Xanh la
        ('Correlogram', 'HSV', 'Random Forest'): '#FF9800',  # Cam
        ('Histogram', 'HSV', 'SVM'): '#F44336',         # Do
        ('Correlogram', 'RGB', 'SVM'): '#9C27B0',      # Tim
    }

    for r in results:
        key = (r['feature'], r['color_space'], r['model'])
        label = f"{r['feature']}\n({r['color_space']})\n+ {r['model']}"
        labels.append(label)
        accuracies.append(r['cv_accuracy'] * 100)
        colors.append(color_map.get(key, '#607D8B'))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), accuracies, color=colors, edgecolor='black',
                  linewidth=0.5, width=0.6)

    # Hien thi gia tri tren moi cot
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('So sanh Accuracy giua cac phuong phap', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Da luu: {save_path}")


def plot_per_class_comparison(results_dict, class_names, save_path):
    """Ve bieu do so sanh accuracy theo tung lop giua Correlogram va Histogram.

    Args:
        results_dict: Dict voi key la ten phuong phap, value la (y_true, y_pred)
        class_names: Ten cac lop
        save_path: Duong dan luu file anh
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(class_names))
    width = 0.35

    for i, (name, (y_true, y_pred)) in enumerate(results_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        offset = (i - len(results_dict)/2 + 0.5) * width
        bars = ax.bar(x + offset, per_class_acc, width, label=name, alpha=0.8)

    ax.set_xlabel('Lop (Class)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('So sanh Accuracy theo tung lop', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Da luu: {save_path}")


def main():
    """Danh gia tat ca mo hinh va tao bieu do."""

    project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_dir = project_dir / "data" / "features"
    models_dir = project_dir / "models"
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DANH GIA MO HINH - COLOR CORRELOGRAM PROJECT")
    print("=" * 60)

    # Tai du lieu
    X_corr_hsv = np.load(features_dir / "correlogram_hsv.npy")
    X_corr_rgb = np.load(features_dir / "correlogram_rgb.npy")
    X_hist_hsv = np.load(features_dir / "histogram_hsv.npy")
    y = np.load(features_dir / "labels.npy")
    class_names = np.load(features_dir / "class_names.npy", allow_pickle=True)

    # Cac thi nghiem can danh gia
    experiments = [
        ("Correlogram_HSV_SVM", X_corr_hsv, "svm_correlogram_hsv.pkl"),
        ("Correlogram_HSV_KNN", X_corr_hsv, "knn_correlogram_hsv.pkl"),
        ("Correlogram_HSV_RF", X_corr_hsv, "rf_correlogram_hsv.pkl"),
        ("Histogram_HSV_SVM", X_hist_hsv, "svm_histogram_hsv.pkl"),
        ("Correlogram_RGB_SVM", X_corr_rgb, "svm_correlogram_rgb.pkl"),
    ]

    all_results = []
    predictions = {}

    for name, X, model_file in experiments:
        print(f"\n--- Danh gia: {name} ---")

        model_path = models_dir / model_file
        if not model_path.exists():
            print(f"  SKIP: Model chua duoc train ({model_file})")
            continue

        model = joblib.load(model_path)

        # Cross-validation predictions
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(model, X, y, cv=cv)

        # Tinh cac chi so
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='macro')
        rec = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')

        print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        # Luu predictions
        predictions[name] = (y, y_pred)

        # Classification report chi tiet
        print(f"\n  Classification Report:")
        report = classification_report(y, y_pred, target_names=class_names)
        print(report)

        # Luu report ra file
        with open(results_dir / f"report_{name}.txt", 'w', encoding='utf-8') as f:
            f.write(f"=== {name} ===\n\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n\n")
            f.write(report)

        # Ve Confusion Matrix
        plot_confusion_matrix(
            y, y_pred, class_names,
            f"Confusion Matrix - {name}",
            results_dir / f"cm_{name}.png"
        )

        all_results.append({
            'name': name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })

    # Ve bieu do so sanh tong hop
    if all_results:
        print("\n--- Tao bieu do so sanh ---")

        # Chuyen doi results cho ham plot
        plot_results = []
        for r in all_results:
            parts = r['name'].split('_')
            if len(parts) >= 3:
                feature = parts[0]
                cs = parts[1]
                model = '_'.join(parts[2:])
            else:
                feature = r['name']
                cs = ''
                model = ''
            plot_results.append({
                'feature': feature,
                'color_space': cs,
                'model': model,
                'cv_accuracy': r['accuracy']
            })

        plot_accuracy_comparison(plot_results, results_dir / "accuracy_comparison.png")

        # So sanh Correlogram vs Histogram theo tung lop
        if "Correlogram_HSV_SVM" in predictions and "Histogram_HSV_SVM" in predictions:
            plot_per_class_comparison(
                {
                    'Correlogram (HSV) + SVM': predictions['Correlogram_HSV_SVM'],
                    'Histogram (HSV) + SVM': predictions['Histogram_HSV_SVM'],
                },
                class_names,
                results_dir / "per_class_comparison.png"
            )

    # Tong ket
    print("\n" + "=" * 60)
    print("TONG KET")
    print("=" * 60)
    print(f"\n{'Phuong phap':<30} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 62)
    for r in all_results:
        print(f"{r['name']:<30} {r['accuracy']:.4f}  {r['precision']:.4f}  "
              f"{r['recall']:.4f}  {r['f1_score']:.4f}")

    # Luu tong ket
    with open(results_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nTat ca ket qua da luu trong: {results_dir}")
    print("\n" + "=" * 60)
    print("DANH GIA HOAN TAT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
