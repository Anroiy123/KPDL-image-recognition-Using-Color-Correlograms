"""
train.py - Huan luyen cac mo hinh hoc may voi split train/test co dinh.

Quy uoc:
- Tune hyperparameter bang cross-validation chi tren train split
- Luu model cuoi tren toan bo train split
- Khong dung test split trong huan luyen
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset_profile import (
    DEFAULT_DATASET_PROFILE,
    list_dataset_profiles,
    resolve_dataset_profile,
    resolve_scoped_or_legacy_path,
    scoped_artifact_name,
)
from dataset_split import load_split_metadata, resolve_split_indices, validate_split_metadata


EXPERIMENTS = [
    {
        'name': 'SpatialCorrelogram_HSV_SVM',
        'feature_key': 'correlogram_hsv_spatial',
        'feature': 'SpatialCorrelogram',
        'color_space': 'HSV',
        'trainer': 'svm',
        'model_file': 'svm_correlogram_hsv_spatial.pkl',
    },
    {
        'name': 'Correlogram_HSV_SVM',
        'feature_key': 'correlogram_hsv',
        'feature': 'Correlogram',
        'color_space': 'HSV',
        'trainer': 'svm',
        'model_file': 'svm_correlogram_hsv.pkl',
    },
    {
        'name': 'Correlogram_HSV_KNN',
        'feature_key': 'correlogram_hsv',
        'feature': 'Correlogram',
        'color_space': 'HSV',
        'trainer': 'knn',
        'model_file': 'knn_correlogram_hsv.pkl',
    },
    {
        'name': 'Correlogram_HSV_RF',
        'feature_key': 'correlogram_hsv',
        'feature': 'Correlogram',
        'color_space': 'HSV',
        'trainer': 'rf',
        'model_file': 'rf_correlogram_hsv.pkl',
    },
    {
        'name': 'Histogram_HSV_SVM',
        'feature_key': 'histogram_hsv',
        'feature': 'Histogram',
        'color_space': 'HSV',
        'trainer': 'svm',
        'model_file': 'svm_histogram_hsv.pkl',
    },
    {
        'name': 'Correlogram_RGB_SVM',
        'feature_key': 'correlogram_rgb',
        'feature': 'Correlogram',
        'color_space': 'RGB',
        'trainer': 'svm',
        'model_file': 'svm_correlogram_rgb.pkl',
    },
]


FEATURE_FILES = {
    'correlogram_hsv': 'correlogram_hsv.npy',
    'correlogram_hsv_spatial': 'correlogram_hsv_spatial.npy',
    'correlogram_rgb': 'correlogram_rgb.npy',
    'histogram_hsv': 'histogram_hsv.npy',
    'histogram_rgb': 'histogram_rgb.npy',
    'labels': 'labels.npy',
    'class_names': 'class_names.npy',
    'image_paths': 'image_paths.npy',
}


def get_project_paths(dataset_profile_key=DEFAULT_DATASET_PROFILE):
    project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_profile = resolve_dataset_profile(dataset_profile_key, project_dir)
    return {
        'project_dir': project_dir,
        'dataset_profile': dataset_profile,
        'data_dir': dataset_profile['data_dir'],
        'features_dir': project_dir / 'data' / 'features',
        'splits_dir': project_dir / 'data' / 'splits',
        'models_dir': project_dir / 'models',
        'results_dir': project_dir / 'results',
    }


def load_features(features_dir, dataset_profile_key):
    """Tai dac trung da trich xuat tu file .npy."""
    features_dir = Path(features_dir)

    resolved_paths = {}
    for key, base_name in FEATURE_FILES.items():
        path, is_legacy = resolve_scoped_or_legacy_path(features_dir, dataset_profile_key, base_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay feature artifact: {path}. "
                f"Hay chay python src/feature_extraction.py --dataset-profile {dataset_profile_key}"
            )
        resolved_paths[key] = path
        if is_legacy:
            print(f"  [legacy] Dang dung artifact cu: {path.name}")

    data = {
        'correlogram_hsv': np.load(resolved_paths['correlogram_hsv']),
        'correlogram_hsv_spatial': np.load(resolved_paths['correlogram_hsv_spatial']),
        'correlogram_rgb': np.load(resolved_paths['correlogram_rgb']),
        'histogram_hsv': np.load(resolved_paths['histogram_hsv']),
        'histogram_rgb': np.load(resolved_paths['histogram_rgb']),
        'labels': np.load(resolved_paths['labels']),
        'class_names': np.load(resolved_paths['class_names'], allow_pickle=True),
        'image_paths': np.load(resolved_paths['image_paths'], allow_pickle=True),
    }

    print('Da tai dac trung:')
    for key, val in data.items():
        print(f'  {key}: {val.shape}')

    return data


def train_svm(X, y, cv=5, n_jobs=-1):
    """Huan luyen SVM voi GridSearchCV tren train split."""
    print('\n--- Huan luyen SVM ---')
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42, probability=True)),
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 0.01, 0.001],
        'svm__kernel': ['rbf', 'linear'],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1,
    )

    start = time.time()
    grid.fit(X, y)
    elapsed = time.time() - start

    print(f'  Tham so tot nhat: {grid.best_params_}')
    print(f'  Accuracy (train CV): {grid.best_score_:.4f}')
    print(f'  Thoi gian: {elapsed:.1f}s')

    return grid.best_estimator_, {
        'model': 'SVM',
        'best_params': grid.best_params_,
        'train_cv_accuracy': float(grid.best_score_),
        'time': elapsed,
    }


def train_knn(X, y, cv=5, n_jobs=-1):
    """Huan luyen KNN voi GridSearchCV tren train split."""
    print('\n--- Huan luyen KNN ---')
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier()),
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan'],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1,
    )

    start = time.time()
    grid.fit(X, y)
    elapsed = time.time() - start

    print(f'  Tham so tot nhat: {grid.best_params_}')
    print(f'  Accuracy (train CV): {grid.best_score_:.4f}')
    print(f'  Thoi gian: {elapsed:.1f}s')

    return grid.best_estimator_, {
        'model': 'KNN',
        'best_params': grid.best_params_,
        'train_cv_accuracy': float(grid.best_score_),
        'time': elapsed,
    }


def train_rf(X, y, cv=5):
    """Huan luyen Random Forest voi GridSearchCV tren train split."""
    print('\n--- Huan luyen Random Forest ---')
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 20, 30],
        'rf__min_samples_split': [2, 5],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    grid.fit(X, y)
    elapsed = time.time() - start

    print(f'  Tham so tot nhat: {grid.best_params_}')
    print(f'  Accuracy (train CV): {grid.best_score_:.4f}')
    print(f'  Thoi gian: {elapsed:.1f}s')

    return grid.best_estimator_, {
        'model': 'Random Forest',
        'best_params': grid.best_params_,
        'train_cv_accuracy': float(grid.best_score_),
        'time': elapsed,
    }


def get_trainer(trainer_name):
    if trainer_name == 'svm':
        return train_svm
    if trainer_name == 'knn':
        return train_knn
    if trainer_name == 'rf':
        return train_rf
    raise ValueError(f'Khong ho tro trainer: {trainer_name}')


def save_model_metadata(meta_path, metadata):
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Huan luyen model theo dataset profile')
    parser.add_argument(
        '--dataset-profile',
        default=DEFAULT_DATASET_PROFILE,
        choices=list_dataset_profiles(),
        help='Dataset profile de train (mac dinh: corel-1k)',
    )
    parser.add_argument(
        '--experiments',
        default='all',
        help='Danh sach ten thi nghiem can train, cach nhau boi dau phay. Mac dinh: all',
    )
    return parser.parse_args()


def main(dataset_profile_key=DEFAULT_DATASET_PROFILE, selected_experiments='all'):
    """Huan luyen tat ca mo hinh va luu ket qua."""
    paths = get_project_paths(dataset_profile_key)
    features_dir = paths['features_dir']
    models_dir = paths['models_dir']
    results_dir = paths['results_dir']
    dataset_profile = paths['dataset_profile']
    split_path = paths['splits_dir'] / dataset_profile['split_filename']

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('HUAN LUYEN MO HINH - COLOR CORRELOGRAM PROJECT')
    print('=' * 60)
    print(f"Dataset profile: {dataset_profile['key']} ({dataset_profile['display_name']})")

    data = load_features(features_dir, dataset_profile['key'])
    if not split_path.exists():
        raise FileNotFoundError(
            f'Khong tim thay split metadata: {split_path}. '
            f'Hay chay python src/feature_extraction.py --dataset-profile {dataset_profile["key"]}'
        )

    split_metadata = load_split_metadata(split_path)
    validate_split_metadata(split_metadata, expected_dataset_name=dataset_profile['dataset_name'])
    split_indices = resolve_split_indices(data['image_paths'], paths['data_dir'], split_metadata)
    train_idx = split_indices['train']
    y = data['labels']

    print('\nSplit dang su dung:')
    for split_name, idx in split_indices.items():
        print(f'  - {split_name}: {len(idx)} mau')

    all_results = []

    experiments_to_run = EXPERIMENTS
    if selected_experiments != 'all':
        requested_names = {name.strip() for name in selected_experiments.split(',') if name.strip()}
        experiments_to_run = [exp for exp in EXPERIMENTS if exp['name'] in requested_names]
        if not experiments_to_run:
            raise ValueError(
                'Khong tim thay thi nghiem nao trung --experiments. '
                f'Ho tro: {", ".join(exp["name"] for exp in EXPERIMENTS)}'
            )
        print(f"\nChi train cac thi nghiem: {', '.join(exp['name'] for exp in experiments_to_run)}")

    for exp_idx, exp in enumerate(experiments_to_run, 1):
        print('\n' + '=' * 60)
        print(f"THI NGHIEM {exp_idx}: {exp['name']}")
        print('=' * 60)

        X = data[exp['feature_key']]
        trainer = get_trainer(exp['trainer'])
        tuned_model, train_info = trainer(X[train_idx], y[train_idx])

        model_path = models_dir / scoped_artifact_name(dataset_profile['key'], exp['model_file'])
        joblib.dump(tuned_model, model_path)

        metadata = {
            'dataset_profile': dataset_profile['key'],
            'dataset_name': dataset_profile['dataset_name'],
            'experiment_name': exp['name'],
            'feature': exp['feature'],
            'feature_key': exp['feature_key'],
            'color_space': exp['color_space'],
            'model': train_info['model'],
            'best_params': {k: str(v) for k, v in train_info['best_params'].items()},
            'split_file': str(split_path),
            'split_counts': {name: int(len(idx)) for name, idx in split_indices.items()},
            'tuning_split': 'train',
            'tuning_method': 'stratified_kfold_cv_on_train',
            'final_training_split': 'train',
            'held_out_test_split': 'test',
            'retrieval_split': 'train',
            'train_cv_accuracy': train_info['train_cv_accuracy'],
        }
        save_model_metadata(models_dir / f'{model_path.stem}.meta.json', metadata)

        result = {
            'experiment_name': exp['name'],
            'model': train_info['model'],
            'feature': exp['feature'],
            'color_space': exp['color_space'],
            'train_cv_accuracy': train_info['train_cv_accuracy'],
            'time': train_info['time'],
            'dataset_profile': dataset_profile['key'],
            'split_file': str(split_path),
            'final_training_split': 'train',
            'best_params': {k: str(v) for k, v in train_info['best_params'].items()},
        }
        all_results.append(result)

    print('\n' + '=' * 60)
    print('TONG KET KET QUA')
    print('=' * 60)
    print(f"\n{'#':<4} {'Feature':<18} {'Color':<6} {'Model':<14} {'TrainCV':<10}")
    print('-' * 63)
    for i, r in enumerate(all_results, 1):
        print(
            f"{i:<4} {r['feature']:<18} {r['color_space']:<6} {r['model']:<14} "
            f"{r['train_cv_accuracy']:.4f}"
        )

    training_results_path = results_dir / scoped_artifact_name(dataset_profile['key'], 'training_results.json')
    with open(training_results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDa luu model vao: {models_dir}")
    print(f"Da luu ket qua vao: {training_results_path}")
    print('\n' + '=' * 60)
    print('HUAN LUYEN HOAN TAT!')
    print('=' * 60)


if __name__ == '__main__':
    args = parse_args()
    main(dataset_profile_key=args.dataset_profile, selected_experiments=args.experiments)
