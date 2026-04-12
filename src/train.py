"""
train.py - Huan luyen cac mo hinh hoc may (SVM, KNN, Random Forest)

Su dung dac trung da trich xuat tu file .npy
Ap dung GridSearchCV de tim tham so tot nhat
Luu model da train vao thu muc models/
"""

import os
import sys
import time
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def load_features(features_dir):
    """Tai dac trung da trich xuat tu file .npy.

    Returns:
        data: Dict chua cac ma tran dac trung va nhan
    """
    features_dir = Path(features_dir)

    data = {
        'correlogram_hsv': np.load(features_dir / "correlogram_hsv.npy"),
        'correlogram_rgb': np.load(features_dir / "correlogram_rgb.npy"),
        'histogram_hsv': np.load(features_dir / "histogram_hsv.npy"),
        'labels': np.load(features_dir / "labels.npy"),
        'class_names': np.load(features_dir / "class_names.npy", allow_pickle=True),
    }

    print("Da tai dac trung:")
    for key, val in data.items():
        print(f"  {key}: {val.shape}")

    return data


def train_svm(X, y, cv=5, n_jobs=-1):
    """Huan luyen SVM voi GridSearchCV.

    Args:
        X: Ma tran dac trung
        y: Vector nhan
        cv: So fold cross-validation
        n_jobs: So process song song cho GridSearchCV

    Returns:
        best_model: Pipeline (Scaler + SVM) tot nhat
        results: Dict ket qua
    """
    print("\n--- Huan luyen SVM ---")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42))
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 0.01, 0.001],
        'svm__kernel': ['rbf', 'linear']
    }

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1
    )

    start = time.time()
    grid.fit(X, y)
    elapsed = time.time() - start

    print(f"  Tham so tot nhat: {grid.best_params_}")
    print(f"  Accuracy (CV): {grid.best_score_:.4f}")
    print(f"  Thoi gian: {elapsed:.1f}s")

    return grid.best_estimator_, {
        'model': 'SVM',
        'best_params': grid.best_params_,
        'cv_accuracy': grid.best_score_,
        'time': elapsed
    }


def train_knn(X, y, cv=5, n_jobs=-1):
    """Huan luyen KNN voi GridSearchCV.

    Returns:
        best_model: Pipeline tot nhat
        results: Dict ket qua
    """
    print("\n--- Huan luyen KNN ---")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1
    )

    start = time.time()
    grid.fit(X, y)
    elapsed = time.time() - start

    print(f"  Tham so tot nhat: {grid.best_params_}")
    print(f"  Accuracy (CV): {grid.best_score_:.4f}")
    print(f"  Thoi gian: {elapsed:.1f}s")

    return grid.best_estimator_, {
        'model': 'KNN',
        'best_params': grid.best_params_,
        'cv_accuracy': grid.best_score_,
        'time': elapsed
    }


def train_rf(X, y, cv=5):
    """Huan luyen Random Forest voi GridSearchCV.

    Returns:
        best_model: Pipeline tot nhat
        results: Dict ket qua
    """
    print("\n--- Huan luyen Random Forest ---")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 20, 30],
        'rf__min_samples_split': [2, 5]
    }

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    start = time.time()
    grid.fit(X, y)
    elapsed = time.time() - start

    print(f"  Tham so tot nhat: {grid.best_params_}")
    print(f"  Accuracy (CV): {grid.best_score_:.4f}")
    print(f"  Thoi gian: {elapsed:.1f}s")

    return grid.best_estimator_, {
        'model': 'Random Forest',
        'best_params': grid.best_params_,
        'cv_accuracy': grid.best_score_,
        'time': elapsed
    }


def main():
    """Huan luyen tat ca mo hinh va luu ket qua."""

    project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_dir = project_dir / "data" / "features"
    models_dir = project_dir / "models"
    results_dir = project_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HUAN LUYEN MO HINH - COLOR CORRELOGRAM PROJECT")
    print("=" * 60)

    # Tai dac trung
    data = load_features(features_dir)
    y = data['labels']
    class_names = data['class_names']

    all_results = []

    # ============================================================
    # THI NGHIEM 1: Correlogram HSV + SVM (KET QUA CHINH)
    # ============================================================
    print("\n" + "=" * 60)
    print("THI NGHIEM 1: Color Correlogram (HSV) + SVM")
    print("=" * 60)
    model_svm, res_svm = train_svm(data['correlogram_hsv'], y)
    res_svm['feature'] = 'Correlogram'
    res_svm['color_space'] = 'HSV'
    all_results.append(res_svm)
    joblib.dump(model_svm, models_dir / "svm_correlogram_hsv.pkl")

    # ============================================================
    # THI NGHIEM 2: Correlogram HSV + KNN
    # ============================================================
    print("\n" + "=" * 60)
    print("THI NGHIEM 2: Color Correlogram (HSV) + KNN")
    print("=" * 60)
    model_knn, res_knn = train_knn(data['correlogram_hsv'], y)
    res_knn['feature'] = 'Correlogram'
    res_knn['color_space'] = 'HSV'
    all_results.append(res_knn)
    joblib.dump(model_knn, models_dir / "knn_correlogram_hsv.pkl")

    # ============================================================
    # THI NGHIEM 3: Correlogram HSV + Random Forest
    # ============================================================
    print("\n" + "=" * 60)
    print("THI NGHIEM 3: Color Correlogram (HSV) + Random Forest")
    print("=" * 60)
    model_rf, res_rf = train_rf(data['correlogram_hsv'], y)
    res_rf['feature'] = 'Correlogram'
    res_rf['color_space'] = 'HSV'
    all_results.append(res_rf)
    joblib.dump(model_rf, models_dir / "rf_correlogram_hsv.pkl")

    # ============================================================
    # THI NGHIEM 4: Histogram HSV + SVM (BASELINE)
    # ============================================================
    print("\n" + "=" * 60)
    print("THI NGHIEM 4: Color Histogram (HSV) + SVM [BASELINE]")
    print("=" * 60)
    model_hist, res_hist = train_svm(data['histogram_hsv'], y)
    res_hist['feature'] = 'Histogram'
    res_hist['color_space'] = 'HSV'
    all_results.append(res_hist)
    joblib.dump(model_hist, models_dir / "svm_histogram_hsv.pkl")

    # ============================================================
    # THI NGHIEM 5: Correlogram RGB + SVM
    # ============================================================
    print("\n" + "=" * 60)
    print("THI NGHIEM 5: Color Correlogram (RGB) + SVM")
    print("=" * 60)
    model_rgb, res_rgb = train_svm(data['correlogram_rgb'], y)
    res_rgb['feature'] = 'Correlogram'
    res_rgb['color_space'] = 'RGB'
    all_results.append(res_rgb)
    joblib.dump(model_rgb, models_dir / "svm_correlogram_rgb.pkl")

    # ============================================================
    # TONG KET
    # ============================================================
    print("\n" + "=" * 60)
    print("TONG KET KET QUA")
    print("=" * 60)
    print(f"\n{'#':<4} {'Feature':<14} {'Color':<6} {'Model':<8} {'CV Acc':<10} {'Time':<8}")
    print("-" * 55)
    for i, r in enumerate(all_results, 1):
        print(f"{i:<4} {r['feature']:<14} {r['color_space']:<6} {r['model']:<8} "
              f"{r['cv_accuracy']:.4f}    {r['time']:.1f}s")

    # Luu ket qua tong hop
    import json
    results_summary = []
    for r in all_results:
        summary = {k: v for k, v in r.items() if k != 'best_params'}
        summary['best_params'] = {k: str(v) for k, v in r['best_params'].items()}
        results_summary.append(summary)

    with open(results_dir / "training_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nDa luu model vao: {models_dir}")
    print(f"Da luu ket qua vao: {results_dir / 'training_results.json'}")

    print("\n" + "=" * 60)
    print("HUAN LUYEN HOAN TAT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
