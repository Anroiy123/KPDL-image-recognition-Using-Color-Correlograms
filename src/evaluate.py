"""
evaluate.py - Danh gia final cac mo hinh tren held-out test split.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dataset_profile import (
    DEFAULT_DATASET_PROFILE,
    list_dataset_profiles,
    resolve_dataset_profile,
    resolve_scoped_or_legacy_path,
    scoped_artifact_name,
)
from dataset_split import load_split_metadata, resolve_split_indices, validate_split_metadata

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


EXPERIMENTS = [
    ('SpatialCorrelogram_HSV_SVM', 'correlogram_hsv_spatial', 'svm_correlogram_hsv_spatial.pkl'),
    ('Correlogram_HSV_SVM', 'correlogram_hsv', 'svm_correlogram_hsv.pkl'),
    ('Correlogram_HSV_KNN', 'correlogram_hsv', 'knn_correlogram_hsv.pkl'),
    ('Histogram_HSV_SVM', 'histogram_hsv', 'svm_histogram_hsv.pkl'),
    ('Correlogram_RGB_SVM', 'correlogram_rgb', 'svm_correlogram_rgb.pkl'),
]


FEATURE_FILES = {
    'correlogram_hsv': 'correlogram_hsv.npy',
    'correlogram_hsv_spatial': 'correlogram_hsv_spatial.npy',
    'correlogram_rgb': 'correlogram_rgb.npy',
    'histogram_hsv': 'histogram_hsv.npy',
    'labels': 'labels.npy',
    'class_names': 'class_names.npy',
    'image_paths': 'image_paths.npy',
}


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Ve confusion matrix dang heatmap."""
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
    print(f'  Da luu: {save_path}')


def plot_accuracy_comparison(results, save_path):
    """Ve bieu do so sanh held-out test accuracy giua cac phuong phap."""
    labels = []
    accuracies = []
    colors = []

    color_map = {
        ('SpatialCorrelogram', 'HSV', 'SVM'): '#1565C0',
        ('Correlogram', 'HSV', 'SVM'): '#2196F3',
        ('Correlogram', 'HSV', 'KNN'): '#4CAF50',
        ('Histogram', 'HSV', 'SVM'): '#F44336',
        ('Correlogram', 'RGB', 'SVM'): '#9C27B0',
    }

    for r in results:
        key = (r['feature'], r['color_space'], r['model'])
        labels.append(f"{r['feature']}\n({r['color_space']})\n+ {r['model']}")
        accuracies.append(r['accuracy'] * 100)
        colors.append(color_map.get(key, '#607D8B'))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), accuracies, color=colors, edgecolor='black', linewidth=0.5, width=0.6)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Held-out Test Accuracy (%)', fontsize=12)
    ax.set_title('So sanh Accuracy tren Tap Test Doc Lap', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Da luu: {save_path}')


def plot_per_class_comparison(results_dict, class_names, save_path):
    """Ve bieu do so sanh accuracy theo lop tren held-out test."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(class_names))
    width = 0.35

    for i, (name, (y_true, y_pred)) in enumerate(results_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        offset = (i - len(results_dict) / 2 + 0.5) * width
        ax.bar(x + offset, per_class_acc, width, label=name, alpha=0.8)

    ax.set_xlabel('Lop (Class)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('So sanh Accuracy Theo Lop Tren Tap Test', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Da luu: {save_path}')


def _parse_experiment_name(name):
    parts = name.split('_')
    if len(parts) >= 3:
        return parts[0], parts[1], '_'.join(parts[2:])
    return name, '', ''


def parse_args():
    parser = argparse.ArgumentParser(description='Danh gia held-out test theo dataset profile')
    parser.add_argument(
        '--dataset-profile',
        default=DEFAULT_DATASET_PROFILE,
        choices=list_dataset_profiles(),
        help='Dataset profile de evaluate (mac dinh: corel-1k)',
    )
    return parser.parse_args()


def main(dataset_profile_key=DEFAULT_DATASET_PROFILE):
    """Danh gia tat ca mo hinh tren held-out test split."""
    project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_profile = resolve_dataset_profile(dataset_profile_key, project_dir)
    data_dir = dataset_profile['data_dir']
    features_dir = project_dir / 'data' / 'features'
    splits_dir = project_dir / 'data' / 'splits'
    models_dir = project_dir / 'models'
    results_dir = project_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    split_path = splits_dir / dataset_profile['split_filename']
    if not split_path.exists():
        raise FileNotFoundError(
            f'Khong tim thay split metadata: {split_path}. '
            f'Hay chay python src/feature_extraction.py --dataset-profile {dataset_profile_key}'
        )

    print('=' * 60)
    print('DANH GIA MO HINH - HELD-OUT TEST')
    print('=' * 60)
    print(f"Dataset profile: {dataset_profile['key']} ({dataset_profile['display_name']})")

    resolved_feature_paths = {}
    for key, base_name in FEATURE_FILES.items():
        path, is_legacy = resolve_scoped_or_legacy_path(features_dir, dataset_profile['key'], base_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay feature artifact: {path}. "
                f"Hay chay python src/feature_extraction.py --dataset-profile {dataset_profile_key}"
            )
        resolved_feature_paths[key] = path
        if is_legacy:
            print(f"  [legacy] Dang dung artifact cu: {path.name}")

    feature_matrices = {
        'correlogram_hsv': np.load(resolved_feature_paths['correlogram_hsv']),
        'correlogram_hsv_spatial': np.load(resolved_feature_paths['correlogram_hsv_spatial']),
        'correlogram_rgb': np.load(resolved_feature_paths['correlogram_rgb']),
        'histogram_hsv': np.load(resolved_feature_paths['histogram_hsv']),
    }
    y = np.load(resolved_feature_paths['labels'])
    class_names = np.load(resolved_feature_paths['class_names'], allow_pickle=True)
    image_paths = np.load(resolved_feature_paths['image_paths'], allow_pickle=True)

    split_metadata = load_split_metadata(split_path)
    validate_split_metadata(split_metadata, expected_dataset_name=dataset_profile['dataset_name'])
    split_indices = resolve_split_indices(image_paths, data_dir, split_metadata)
    test_idx = split_indices['test']
    y_test = y[test_idx]

    print('\nSplit dang su dung:')
    for split_name, idx in split_indices.items():
        print(f'  - {split_name}: {len(idx)} mau')

    all_results = []
    predictions = {}
    result_prefix = dataset_profile['key']

    for name, feature_key, model_file in EXPERIMENTS:
        print(f'\n--- Danh gia: {name} ---')
        model_path, is_legacy_model = resolve_scoped_or_legacy_path(models_dir, dataset_profile['key'], model_file)
        if not model_path.exists():
            print(f'  SKIP: Model chua duoc train ({model_file})')
            continue
        if is_legacy_model:
            print(f'  [legacy] Dang dung model cu: {model_path.name}')

        X_test = feature_matrices[feature_key][test_idx]
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f'  Held-out test accuracy:  {acc:.4f} ({acc*100:.1f}%)')
        print(f'  Precision: {prec:.4f}')
        print(f'  Recall:    {rec:.4f}')
        print(f'  F1-Score:  {f1:.4f}')

        predictions[name] = (y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

        report_path = results_dir / f'report_{result_prefix}_{name}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f'=== {name} ===\n\n')
            f.write(f'Dataset profile: {dataset_profile["key"]}\n')
            f.write(f'Evaluation split: test\n')
            f.write(f'Split file: {split_path}\n')
            f.write(f'Accuracy:  {acc:.4f}\n')
            f.write(f'Precision: {prec:.4f}\n')
            f.write(f'Recall:    {rec:.4f}\n')
            f.write(f'F1-Score:  {f1:.4f}\n\n')
            f.write(report)

        plot_confusion_matrix(
            y_test,
            y_pred,
            class_names,
            f'Confusion Matrix - {name} (Held-out Test)',
            results_dir / f'cm_{result_prefix}_{name}.png',
        )

        feature_name, color_space, model_name = _parse_experiment_name(name)
        all_results.append({
            'name': name,
            'feature': feature_name,
            'color_space': color_space,
            'model': model_name,
            'dataset_profile': dataset_profile['key'],
            'dataset_name': dataset_profile['dataset_name'],
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'evaluation_split': 'test',
            'split_file': str(split_path),
        })

    if all_results:
        print('\n--- Tao bieu do so sanh ---')
        plot_accuracy_comparison(all_results, results_dir / f'accuracy_comparison_{result_prefix}.png')

        if 'SpatialCorrelogram_HSV_SVM' in predictions and 'Histogram_HSV_SVM' in predictions:
            plot_per_class_comparison(
                {
                    'Spatial Correlogram (HSV) + SVM': predictions['SpatialCorrelogram_HSV_SVM'],
                    'Histogram (HSV) + SVM': predictions['Histogram_HSV_SVM'],
                },
                class_names,
                results_dir / f'per_class_comparison_{result_prefix}.png',
            )

    print('\n' + '=' * 60)
    print('TONG KET')
    print('=' * 60)
    print(f"\n{'Phuong phap':<30} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print('-' * 62)
    for r in all_results:
        print(f"{r['name']:<30} {r['accuracy']:.4f}  {r['precision']:.4f}  {r['recall']:.4f}  {r['f1_score']:.4f}")

    summary_path = results_dir / scoped_artifact_name(dataset_profile['key'], 'evaluation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f'\nTat ca ket qua da luu trong: {results_dir}')
    print(f'Summary: {summary_path}')
    print('\n' + '=' * 60)
    print('DANH GIA HOAN TAT!')
    print('=' * 60)


if __name__ == '__main__':
    args = parse_args()
    main(dataset_profile_key=args.dataset_profile)
