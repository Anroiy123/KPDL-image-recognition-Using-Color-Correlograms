# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- Install dependencies: `pip install -r requirements.txt`
- Extract features from `data/corel-1k/`: `python src/feature_extraction.py`
- Train all planned models: `python src/train.py`
- Evaluate trained models and regenerate reports/charts: `python src/evaluate.py`
- Run the demo app: `streamlit run app.py`
- Open notebooks: `cd notebooks && jupyter notebook`
- Quick algorithm self-tests:
  - `python src/color_correlogram.py`
  - `python src/color_histogram.py`

There is no dedicated build, lint, or unit test framework in this repo. The practical validation flow is:
1. `python src/feature_extraction.py`
2. `python src/train.py`
3. `python src/evaluate.py`
4. `streamlit run app.py`

## Architecture Overview

This repo is a small ML pipeline for image classification using Color Correlogram features on the Corel-1K dataset.

End-to-end flow:
1. `src/preprocessing.py` loads images from `data/corel-1k/`, resizes them to `128x128`, converts color spaces, and quantizes colors.
2. `src/color_correlogram.py` computes the main feature descriptor (auto-correlogram). `src/color_histogram.py` provides the baseline descriptor for comparison.
3. `src/feature_extraction.py` runs preprocessing + feature extraction over the full dataset, writes reusable `.npy` artifacts into `data/features/`, and generates split metadata in `data/splits/`.
4. `src/train.py` loads those saved features, tunes on the `train` split, checks quality on `val`, refits final models on `train+val`, and saves `.pkl` models plus `.meta.json` provenance into `models/`.
5. `src/evaluate.py` reloads features, split metadata, and models, then computes final metrics only on the held-out `test` split and writes reports/plots into `results/`.
6. `app.py` is a Streamlit demo that loads the spatial HSV SVM model plus saved features/image paths to classify uploaded images and optionally retrieve similar images from the non-test split.

The notebooks mirror this same pipeline for exploration and presentation:
- `01_kham_pha_du_lieu.ipynb`: dataset exploration
- `02_color_correlogram_demo.ipynb`: feature demonstration
- `03_huan_luyen_danh_gia.ipynb`: training/evaluation walkthrough
- `04_so_sanh_ket_qua.ipynb`: result comparison

## Project-Specific Patterns

- Dataset layout matters. The code expects class-per-folder data under `data/corel-1k/`. Current class names are: `africans`, `beaches`, `buildings`, `buses`, `dinosaurs`, `elephants`, `flowers`, `food`, `horses`, `mountains`.
- Default image size is fixed at `128x128` in preprocessing. Keep this consistent unless you intend to regenerate all downstream artifacts.
- The primary pipeline is **HSV Color Correlogram + SVM**. Other configurations exist mainly for comparison:
  - HSV correlogram: main feature
  - RGB correlogram: comparison variant
  - HSV histogram: baseline
- Default correlogram settings are repo-significant:
  - HSV quantization: `8 x 3 x 3 = 72` colors
  - Distances: `{1, 3, 5, 7}`
  - Main demo feature vector size: `1440` (`spatial correlogram`)
- `src/color_correlogram.py` contains both a slower reference implementation and the faster vectorized implementation. Use `auto_correlogram_fast` for normal pipeline work.
- Color quantization code explicitly casts image channels to `int32` before arithmetic to avoid overflow on `uint8`. Preserve that behavior if you edit quantization logic.
- `src/train.py` assumes feature files already exist in `data/features/`; it does not regenerate them.
- `src/evaluate.py` assumes trained models and split metadata already exist; it does not retrain them.
- `app.py` is tightly coupled to the trained spatial HSV SVM model (`models/svm_correlogram_hsv_spatial.pkl`) and the saved spatial HSV feature artifacts in `data/features/`. If you change feature dimensions, split policy, or defaults, retrain and regenerate artifacts before expecting the app to work.
- In the Streamlit sidebar, changing color space or bin settings can make uploaded-image features incompatible with the saved model. The current stable demo configuration is HSV with `H=8`, `S=3`, `V=3`.

Important generated artifacts:
- `data/features/`
  - `correlogram_hsv.npy`
  - `correlogram_hsv_spatial.npy`
  - `correlogram_rgb.npy`
  - `histogram_hsv.npy`
  - `histogram_rgb.npy`
  - `labels.npy`
  - `class_names.npy`
  - `image_paths.npy`
- `data/splits/`
  - `corel-1k_split.json`
- `models/`
  - `svm_correlogram_hsv_spatial.pkl`
  - `svm_correlogram_hsv.pkl`
  - `knn_correlogram_hsv.pkl`
  - `rf_correlogram_hsv.pkl`
  - `svm_histogram_hsv.pkl`
  - `svm_correlogram_rgb.pkl`
  - `*.meta.json`
- `results/`
  - `training_results.json`
  - `evaluation_summary.json`
  - `report_*.txt`
  - `cm_*.png`
  - `accuracy_comparison.png`
  - `per_class_comparison.png`

If you change preprocessing, quantization, feature dimensions, split policy, class names, or dataset layout, expect to regenerate `data/features/`, refresh `data/splits/`, retrain models in `models/`, and rerun evaluation to refresh `results/`.
