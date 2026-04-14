"""
experiment_runner.py - Chay thuc nghiem linh hoat theo cau hinh.

Vi du:
python src/experiment_runner.py --feature correlogram --color hsv --model svm --eval stratified_holdout
python src/experiment_runner.py --feature histogram --color rgb --model knn --eval kfold --k 5
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from evaluate import plot_confusion_matrix
from dataset_split import (
    DEFAULT_SPLIT_FILENAME,
    load_split_metadata,
    merge_split_indices,
    resolve_split_indices,
)
from evaluation_methods import (
    evaluate_bootstrap,
    evaluate_holdout,
    evaluate_kfold,
    evaluate_leave_one_out,
    evaluate_repeated_holdout,
    evaluate_stratified_holdout,
)
from feature_extraction import extract_all_features
from preprocessing import load_dataset
from train import train_knn, train_svm


FEATURE_FILE_MAP = {
    ('correlogram', 'hsv'): 'correlogram_hsv.npy',
    ('spatial_correlogram', 'hsv'): 'correlogram_hsv_spatial.npy',
    ('correlogram', 'rgb'): 'correlogram_rgb.npy',
    ('histogram', 'hsv'): 'histogram_hsv.npy',
    ('histogram', 'rgb'): 'histogram_rgb.npy',
}

EVAL_LABELS = {
    'independent_test': 'Independent Held-out Test',
    'holdout': 'Hold-out',
    'stratified_holdout': 'Stratified Hold-out',
    'repeated_holdout': 'Repeated Hold-out',
    'kfold': 'k-Fold Cross-Validation',
    'leave_one_out': 'Leave-One-Out',
    'bootstrap': 'Bootstrap',
}


def get_project_paths():
    project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {
        'project_dir': project_dir,
        'data_dir': project_dir / 'data' / 'corel-1k',
        'features_dir': project_dir / 'data' / 'features',
        'splits_dir': project_dir / 'data' / 'splits',
        'results_dir': project_dir / 'results',
        'models_dir': project_dir / 'models',
    }


def load_feature_matrix(feature_method, color_space):
    """Tai feature matrix, neu thieu thi tu tao tu dataset."""
    paths = get_project_paths()
    features_dir = paths['features_dir']
    features_dir.mkdir(parents=True, exist_ok=True)
    key = (feature_method, color_space)
    if key not in FEATURE_FILE_MAP:
        raise ValueError(f'Khong ho tro ket hop feature/color: {feature_method}/{color_space}')
    feature_file = features_dir / FEATURE_FILE_MAP[key]

    if not feature_file.exists():
        images, _, _ = load_dataset(paths['data_dir'])
        if len(images) == 0:
            raise FileNotFoundError('Khong tim thay dataset de tao feature file.')

        if feature_method == 'spatial_correlogram':
            X = extract_all_features(
                images,
                method='correlogram',
                color_space=color_space,
                spatial_grid=2,
                include_global=True,
            )
        else:
            X = extract_all_features(images, method=feature_method, color_space=color_space)
        np.save(feature_file, X)
    else:
        X = np.load(feature_file)

    labels_path = features_dir / 'labels.npy'
    class_names_path = features_dir / 'class_names.npy'
    image_paths_path = features_dir / 'image_paths.npy'
    if not labels_path.exists() or not class_names_path.exists() or not image_paths_path.exists():
        raise FileNotFoundError('Thieu labels.npy, class_names.npy hoac image_paths.npy. Hay chay python src/feature_extraction.py')

    y = np.load(labels_path)
    class_names = np.load(class_names_path, allow_pickle=True)
    image_paths = np.load(image_paths_path, allow_pickle=True)
    return X, y, class_names, image_paths, feature_file


def _basic_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def get_train_model_fn(model_name):
    """Tra ve ham train co chung interface train_model_fn(X_train, y_train)."""
    if model_name == 'svm':
        return lambda X_train, y_train: train_svm(X_train, y_train, cv=5, n_jobs=1)
    if model_name == 'knn':
        return lambda X_train, y_train: train_knn(X_train, y_train, cv=5, n_jobs=1)
    raise ValueError(f'Model khong ho tro: {model_name}')


def get_cv_model(model_name):
    """Mo hinh co hyperparameter co dinh de su dung cho CV methods."""
    if model_name == 'svm':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=10, gamma='scale', kernel='rbf', random_state=42)),
        ])
    if model_name == 'knn':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')),
        ])
    raise ValueError(f'Model khong ho tro: {model_name}')


def run_experiment(feature, color, model_name, eval_method, test_size=0.2, k=5, n_repeats=10, n_iterations=30, sample_ratio=0.8, random_state=42):
    X, y, class_names, image_paths, feature_file = load_feature_matrix(feature, color)
    train_model_fn = get_train_model_fn(model_name)
    paths = get_project_paths()

    if eval_method == 'independent_test':
        split_path = paths['splits_dir'] / DEFAULT_SPLIT_FILENAME
        if not split_path.exists():
            raise FileNotFoundError(f'Thieu split metadata: {split_path}. Hay chay python src/feature_extraction.py')

        split_metadata = load_split_metadata(split_path)
        split_indices = resolve_split_indices(image_paths, paths['data_dir'], split_metadata)
        train_idx = split_indices['train']
        val_idx = split_indices['val']
        test_idx = split_indices['test']
        train_val_idx = merge_split_indices(split_indices, ['train', 'val'])

        tuned_model, train_info = train_model_fn(X[train_idx], y[train_idx])
        y_val_pred = tuned_model.predict(X[val_idx])
        val_summary = _basic_metrics(y[val_idx], y_val_pred)

        final_model = clone(tuned_model)
        final_model.fit(X[train_val_idx], y[train_val_idx])

        y_true = y[test_idx]
        y_pred = final_model.predict(X[test_idx])
        eval_info = {
            'method': 'independent_test',
            'split_file': str(split_path),
            'split_counts': {name: int(len(idx)) for name, idx in split_indices.items()},
            'training': train_info,
            'validation_summary': val_summary,
            'final_training_split': 'train+val',
            'summary': _basic_metrics(y_true, y_pred),
        }
    elif eval_method == 'holdout':
        y_true, y_pred, eval_info = evaluate_holdout(train_model_fn, X, y, test_size=test_size, random_state=random_state)
    elif eval_method == 'stratified_holdout':
        y_true, y_pred, eval_info = evaluate_stratified_holdout(train_model_fn, X, y, test_size=test_size, random_state=random_state)
    elif eval_method == 'repeated_holdout':
        y_true, y_pred, eval_info = evaluate_repeated_holdout(train_model_fn, X, y, test_size=test_size, n_repeats=n_repeats, stratify=True, random_state=random_state)
    elif eval_method == 'kfold':
        y_true, y_pred, eval_info = evaluate_kfold(get_cv_model(model_name), X, y, n_splits=k, stratified=True, random_state=random_state)
    elif eval_method == 'leave_one_out':
        y_true, y_pred, eval_info = evaluate_leave_one_out(get_cv_model(model_name), X, y)
    elif eval_method == 'bootstrap':
        y_true, y_pred, eval_info = evaluate_bootstrap(train_model_fn, X, y, n_iterations=n_iterations, sample_ratio=sample_ratio, random_state=random_state)
    else:
        raise ValueError(f'Phuong phap danh gia khong ho tro: {eval_method}')

    summary = dict(eval_info['summary'])
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'{feature}_{color}_{model_name}_{eval_method}'
    results_dir = paths['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_path = results_dir / f'cm_{experiment_name}_{timestamp}.png'
    json_path = results_dir / f'experiment_{experiment_name}_{timestamp}.json'
    report_path = results_dir / f'report_{experiment_name}_{timestamp}.txt'

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        f'{feature.title()} + {color.upper()} + {model_name.upper()} ({EVAL_LABELS[eval_method]})',
        cm_path,
    )

    result = {
        'experiment_name': experiment_name,
        'feature': feature,
        'color_space': color,
        'model': model_name,
        'evaluation_method': eval_method,
        'evaluation_label': EVAL_LABELS[eval_method],
        'feature_file': str(feature_file),
        'dataset_shape': [int(X.shape[0]), int(X.shape[1])],
        'evaluation_dataset_shape': [int(len(y_true)), int(X.shape[1])],
        'class_count': int(len(class_names)),
        'summary': summary,
        'evaluation_details': eval_info,
        'classification_report': report_dict,
        'artifacts': {
            'confusion_matrix': str(cm_path),
            'json': str(json_path),
            'report': str(report_path),
        },
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with open(report_path, 'w', encoding='utf-8') as f:
        if eval_method == 'independent_test':
            f.write(f"Validation summary: {json.dumps(eval_info['validation_summary'], ensure_ascii=False)}\n")
            f.write(f"Split file: {eval_info['split_file']}\n\n")
        f.write(report_text)

    return result, report_text


def print_summary(result):
    summary = result['summary']
    print('=' * 70)
    print('KET QUA THI NGHIEM')
    print('=' * 70)
    print(f"Feature:     {result['feature']}")
    print(f"Color space: {result['color_space']}")
    print(f"Model:       {result['model']}")
    print(f"Evaluation:  {result['evaluation_label']}")
    print(f"Accuracy:    {summary['accuracy']:.4f}")
    if 'accuracy_std' in summary:
        print(f"Accuracy SD: {summary['accuracy_std']:.4f}")
    print(f"Precision:   {summary['precision']:.4f}")
    print(f"Recall:      {summary['recall']:.4f}")
    print(f"F1-score:    {summary['f1_score']:.4f}")
    print(f"JSON:        {result['artifacts']['json']}")
    print(f"Report:      {result['artifacts']['report']}")
    print(f"Confusion:   {result['artifacts']['confusion_matrix']}")


def parse_args():
    parser = argparse.ArgumentParser(description='Chay thi nghiem linh hoat cho do an Color Correlogram')
    parser.add_argument('--feature', choices=['correlogram', 'spatial_correlogram', 'histogram'], required=True)
    parser.add_argument('--color', choices=['hsv', 'rgb'], required=True)
    parser.add_argument('--model', choices=['svm', 'knn'], required=True)
    parser.add_argument('--eval', choices=['independent_test', 'holdout', 'stratified_holdout', 'repeated_holdout', 'kfold', 'leave_one_out', 'bootstrap'], required=True)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--n-iterations', type=int, default=30)
    parser.add_argument('--sample-ratio', type=float, default=0.8)
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    result, report_text = run_experiment(
        feature=args.feature,
        color=args.color,
        model_name=args.model,
        eval_method=args.eval,
        test_size=args.test_size,
        k=args.k,
        n_repeats=args.n_repeats,
        n_iterations=args.n_iterations,
        sample_ratio=args.sample_ratio,
        random_state=args.random_state,
    )
    print_summary(result)
    print('\nClassification report:\n')
    print(report_text)


if __name__ == '__main__':
    main()
