"""
evaluation_methods.py - Cac phuong phap danh gia mo hinh linh hoat.

Cac ham trong file nay phuc vu cho experiment runner:
- Hold-out
- Stratified Hold-out
- Repeated Hold-out
- k-Fold Cross-Validation
- Leave-One-Out
- Bootstrap sampling
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)


def _basic_metrics(y_true, y_pred):
    """Tinh cac metric co ban va tra ve dict serializable."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def evaluate_holdout(train_model_fn, X, y, test_size=0.2, random_state=42):
    """Danh gia bang hold-out khong stratify."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    model, train_info = train_model_fn(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred, {
        'method': 'holdout',
        'test_size': test_size,
        'random_state': random_state,
        'split_sizes': {
            'train': int(len(y_train)),
            'test': int(len(y_test)),
        },
        'training': train_info,
        'summary': _basic_metrics(y_test, y_pred),
    }


def evaluate_stratified_holdout(train_model_fn, X, y, test_size=0.2, random_state=42):
    """Danh gia bang hold-out co stratify theo nhan."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )
    model, train_info = train_model_fn(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred, {
        'method': 'stratified_holdout',
        'test_size': test_size,
        'random_state': random_state,
        'split_sizes': {
            'train': int(len(y_train)),
            'test': int(len(y_test)),
        },
        'training': train_info,
        'summary': _basic_metrics(y_test, y_pred),
    }


def evaluate_repeated_holdout(train_model_fn, X, y, test_size=0.2, n_repeats=10, stratify=True, random_state=42):
    """Danh gia bang repeated hold-out, tong hop metric theo nhieu lan chia."""
    all_true = []
    all_pred = []
    repeat_metrics = []
    training_runs = []

    for i in range(n_repeats):
        repeat_state = random_state + i
        stratify_labels = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=repeat_state,
            shuffle=True,
            stratify=stratify_labels,
        )
        model, train_info = train_model_fn(X_train, y_train)
        y_pred = model.predict(X_test)

        all_true.append(y_test)
        all_pred.append(y_pred)
        repeat_metrics.append({
            'repeat': i + 1,
            'random_state': repeat_state,
            **_basic_metrics(y_test, y_pred),
        })
        training_runs.append(train_info)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    accuracies = [item['accuracy'] for item in repeat_metrics]
    precisions = [item['precision'] for item in repeat_metrics]
    recalls = [item['recall'] for item in repeat_metrics]
    f1_scores = [item['f1_score'] for item in repeat_metrics]

    return y_true, y_pred, {
        'method': 'repeated_holdout',
        'test_size': test_size,
        'n_repeats': n_repeats,
        'stratified': bool(stratify),
        'random_state': random_state,
        'training_runs': training_runs,
        'repeat_metrics': repeat_metrics,
        'summary': {
            'accuracy': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'precision': float(np.mean(precisions)),
            'precision_std': float(np.std(precisions)),
            'recall': float(np.mean(recalls)),
            'recall_std': float(np.std(recalls)),
            'f1_score': float(np.mean(f1_scores)),
            'f1_score_std': float(np.std(f1_scores)),
        },
    }


def evaluate_kfold(model, X, y, n_splits=5, stratified=True, random_state=42):
    """Danh gia bang k-fold CV voi mo hinh da co hyperparameter co dinh."""
    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_pred = cross_val_predict(clone(model), X, y, cv=cv)
    return y, y_pred, {
        'method': 'kfold',
        'n_splits': n_splits,
        'stratified': bool(stratified),
        'random_state': random_state,
        'summary': _basic_metrics(y, y_pred),
    }


def evaluate_leave_one_out(model, X, y):
    """Danh gia bang Leave-One-Out CV."""
    cv = LeaveOneOut()
    y_pred = cross_val_predict(clone(model), X, y, cv=cv)

    return y, y_pred, {
        'method': 'leave_one_out',
        'n_splits': int(len(y)),
        'summary': _basic_metrics(y, y_pred),
    }


def evaluate_bootstrap(train_model_fn, X, y, n_iterations=30, sample_ratio=0.8, random_state=42):
    """Danh gia bang bootstrap sampling voi out-of-bag test set."""
    rng = np.random.default_rng(random_state)
    n_samples = len(y)
    sample_size = max(1, int(n_samples * sample_ratio))

    all_true = []
    all_pred = []
    iteration_metrics = []
    training_runs = []
    skipped_iterations = 0

    for i in range(n_iterations):
        train_indices = rng.choice(n_samples, size=sample_size, replace=True)
        unique_train = np.unique(train_indices)
        test_mask = np.ones(n_samples, dtype=bool)
        test_mask[unique_train] = False
        test_indices = np.flatnonzero(test_mask)

        if len(test_indices) == 0:
            skipped_iterations += 1
            continue

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        model, train_info = train_model_fn(X_train, y_train)
        y_pred = model.predict(X_test)

        all_true.append(y_test)
        all_pred.append(y_pred)
        iteration_metrics.append({
            'iteration': i + 1,
            'oob_size': int(len(test_indices)),
            **_basic_metrics(y_test, y_pred),
        })
        training_runs.append(train_info)

    if not all_true:
        raise ValueError('Khong tao duoc out-of-bag test set hop le cho bootstrap.')

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    accuracies = [item['accuracy'] for item in iteration_metrics]
    precisions = [item['precision'] for item in iteration_metrics]
    recalls = [item['recall'] for item in iteration_metrics]
    f1_scores = [item['f1_score'] for item in iteration_metrics]

    return y_true, y_pred, {
        'method': 'bootstrap',
        'n_iterations': n_iterations,
        'sample_ratio': sample_ratio,
        'random_state': random_state,
        'successful_iterations': int(len(iteration_metrics)),
        'skipped_iterations': int(skipped_iterations),
        'training_runs': training_runs,
        'iteration_metrics': iteration_metrics,
        'summary': {
            'accuracy': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'precision': float(np.mean(precisions)),
            'precision_std': float(np.std(precisions)),
            'recall': float(np.mean(recalls)),
            'recall_std': float(np.std(recalls)),
            'f1_score': float(np.mean(f1_scores)),
            'f1_score_std': float(np.std(f1_scores)),
        },
    }
