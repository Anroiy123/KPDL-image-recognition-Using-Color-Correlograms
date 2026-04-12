"""
app.py - Ung dung demo Streamlit cho Color Correlogram Image Recognition

Chay: streamlit run app.py
"""

import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st

# Them duong dan src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from color_correlogram import extract_correlogram_feature
from experiment_runner import EVAL_LABELS, run_experiment


PROJECT_DIR = Path(__file__).parent
FEATURES_DIR = PROJECT_DIR / "data" / "features"
MODELS_DIR = PROJECT_DIR / "models"

MODEL_CONFIGS = {
    'hsv': {
        'label': 'HSV',
        'model_file': 'svm_correlogram_hsv.pkl',
        'features_file': 'correlogram_hsv.npy',
        'sidebar_message': 'Model HSV duoc train co dinh voi H=8, S=3, V=3.',
        'h_bins': 8,
        's_bins': 3,
        'v_bins': 3,
        'rgb_bins': 4,
    },
    'rgb': {
        'label': 'RGB',
        'model_file': 'svm_correlogram_rgb.pkl',
        'features_file': 'correlogram_rgb.npy',
        'sidebar_message': 'Model RGB duoc train co dinh voi RGB bins = 4.',
        'h_bins': 8,
        's_bins': 3,
        'v_bins': 3,
        'rgb_bins': 4,
    }
}


def load_class_names():
    """Tai danh sach ten lop."""
    class_names_path = FEATURES_DIR / "class_names.npy"
    if not class_names_path.exists():
        return None
    return np.load(class_names_path, allow_pickle=True)


def load_model_and_data(color_space):
    """Tai model va du lieu da luu theo khong gian mau da chon."""
    config = MODEL_CONFIGS[color_space]
    model_path = MODELS_DIR / config['model_file']
    if not model_path.exists():
        return None, None, None

    class_names = load_class_names()
    if class_names is None:
        return None, None, None

    model = joblib.load(model_path)
    paths_file = FEATURES_DIR / "image_paths.npy"
    features_file = FEATURES_DIR / config['features_file']

    image_paths = None
    db_features = None
    if paths_file.exists() and features_file.exists():
        image_paths = np.load(paths_file, allow_pickle=True)
        db_features = np.load(features_file)

    return model, class_names, (image_paths, db_features)


def find_similar_images(query_feat, db_features, image_paths, top_k=3):
    """Tim cac anh tuong tu trong database."""
    from scipy.spatial.distance import cdist

    distances = cdist([query_feat], db_features, metric='cosine')[0]
    indices = np.argsort(distances)[:top_k]
    return [(image_paths[i], distances[i]) for i in indices]


def render_summary_metrics(summary):
    """Hien thi cac metric tong hop."""
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{summary['accuracy'] * 100:.2f}%")
    cols[1].metric("Precision", f"{summary['precision']:.4f}")
    cols[2].metric("Recall", f"{summary['recall']:.4f}")
    cols[3].metric("F1-score", f"{summary['f1_score']:.4f}")

    if 'accuracy_std' in summary:
        st.caption(
            f"Do lech chuan: acc={summary['accuracy_std']:.4f}, "
            f"precision={summary.get('precision_std', 0.0):.4f}, "
            f"recall={summary.get('recall_std', 0.0):.4f}, "
            f"f1={summary.get('f1_score_std', 0.0):.4f}"
        )


def render_prediction_tab():
    """Tab nhan dang anh bang model da train san."""
    st.subheader("🎯 Nhan dang anh")
    st.markdown("Chon khong gian mau dung voi model da train, sau do upload anh de du doan.")

    color_space_label = st.selectbox("Khong gian mau", ["HSV", "RGB"], key="predict_color_space")
    color_space = color_space_label.lower()
    config = MODEL_CONFIGS[color_space]

    config_col1, config_col2, config_col3 = st.columns(3)
    if color_space == 'hsv':
        config_col1.number_input("H bins", value=config['h_bins'], step=1, disabled=True, key="predict_h_bins")
        config_col2.number_input("S bins", value=config['s_bins'], step=1, disabled=True, key="predict_s_bins")
        config_col3.number_input("V bins", value=config['v_bins'], step=1, disabled=True, key="predict_v_bins")
    else:
        config_col1.number_input("RGB bins", value=config['rgb_bins'], step=1, disabled=True, key="predict_rgb_bins")

    st.caption(config['sidebar_message'])

    model, class_names, db_data = load_model_and_data(color_space)
    if model is None:
        st.error("❌ Chua co model hoac feature files can thiet. Hay chay:")
        st.code("""
pip install -r requirements.txt
python src/feature_extraction.py
python src/train.py
streamlit run app.py
        """)
        return

    with st.expander("Danh sach cac lop nhan dang"):
        for i, name in enumerate(class_names, 1):
            st.write(f"{i}. {name}")

    uploaded_file = st.file_uploader(
        "📁 Upload anh can nhan dang",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key="predict_upload",
    )

    if uploaded_file is None:
        st.info("👆 Hay upload 1 anh de bat dau nhan dang!")
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Anh goc")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=300)

    with st.spinner("Dang trich xuat dac trung..."):
        feat = extract_correlogram_feature(
            img,
            color_space=color_space,
            h_bins=config['h_bins'],
            s_bins=config['s_bins'],
            v_bins=config['v_bins'],
            rgb_bins=config['rgb_bins'],
        )

    with col2:
        st.subheader("📊 Vector dac trung Correlogram")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(feat)), feat, color='coral', width=1, alpha=0.7)
        ax.set_xlabel('Chieu (dimension)')
        ax.set_ylabel('Gia tri')
        ax.set_title(f"Color Correlogram ({config['label']})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Kết quả dự đoán")

    with st.spinner("Dang du doan..."):
        prediction = model.predict([feat])
        predicted_class = class_names[prediction[0]]

    st.success(f"**Lop du doan: {predicted_class}**")

    if db_data[0] is not None and db_data[1] is not None:
        st.subheader("🔍 Anh tuong tu trong database")
        try:
            similar = find_similar_images(feat, db_data[1], db_data[0], top_k=5)
            cols = st.columns(5)
            for i, (path, dist) in enumerate(similar):
                if os.path.exists(path):
                    sim_img = cv2.imread(path)
                    sim_img = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
                    with cols[i]:
                        st.image(sim_img, caption=f"d={dist:.3f}", width=120)
        except Exception as e:
            st.warning(f"Khong the tim anh tuong tu: {e}")


def render_experiment_table(report_dict):
    """Hien thi bang metric theo tung lop."""
    rows = []
    for label, values in report_dict.items():
        if not isinstance(values, dict):
            continue
        if label in ('macro avg', 'weighted avg'):
            continue
        rows.append({
            'class': label,
            'precision': round(values['precision'], 4),
            'recall': round(values['recall'], 4),
            'f1-score': round(values['f1-score'], 4),
            'support': int(values['support']),
        })

    if rows:
        st.dataframe(rows, use_container_width=True)

    macro_avg = report_dict.get('macro avg')
    weighted_avg = report_dict.get('weighted avg')
    if isinstance(macro_avg, dict) and isinstance(weighted_avg, dict):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Macro avg**")
            st.write({
                'precision': round(macro_avg['precision'], 4),
                'recall': round(macro_avg['recall'], 4),
                'f1-score': round(macro_avg['f1-score'], 4),
                'support': int(macro_avg['support']),
            })
        with col2:
            st.write("**Weighted avg**")
            st.write({
                'precision': round(weighted_avg['precision'], 4),
                'recall': round(weighted_avg['recall'], 4),
                'f1-score': round(weighted_avg['f1-score'], 4),
                'support': int(weighted_avg['support']),
            })


def render_evaluation_controls():
    """Lay cau hinh thi nghiem tu UI."""
    top_cols = st.columns(4)
    feature = top_cols[0].selectbox("Feature", ['correlogram', 'histogram'], key="eval_feature")
    color = top_cols[1].selectbox("Color space", ['hsv', 'rgb'], key="eval_color")
    model_name = top_cols[2].selectbox("Model", ['svm', 'knn'], key="eval_model")
    eval_method = top_cols[3].selectbox(
        "Evaluation",
        list(EVAL_LABELS.keys()),
        format_func=lambda key: EVAL_LABELS[key],
        key="eval_method",
    )

    params = {
        'test_size': 0.2,
        'k': 5,
        'n_repeats': 10,
        'n_iterations': 30,
        'sample_ratio': 0.8,
        'random_state': 42,
    }

    param_cols = st.columns(3)
    if eval_method in ('holdout', 'stratified_holdout', 'repeated_holdout'):
        params['test_size'] = param_cols[0].slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    if eval_method == 'kfold':
        params['k'] = int(param_cols[0].number_input("So fold (k)", min_value=2, max_value=10, value=5, step=1))
    if eval_method == 'repeated_holdout':
        params['n_repeats'] = int(param_cols[1].number_input("So lan lap", min_value=2, max_value=30, value=10, step=1))
    if eval_method == 'bootstrap':
        params['n_iterations'] = int(param_cols[0].number_input("So bootstrap iterations", min_value=5, max_value=100, value=30, step=5))
        params['sample_ratio'] = param_cols[1].slider("Ti le mau bootstrap", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
    if eval_method == 'leave_one_out':
        st.warning("Leave-One-Out rat cham vi phai train lai gan bang so luong anh trong dataset.")

    params['random_state'] = int(param_cols[2].number_input("Random state", min_value=0, max_value=9999, value=42, step=1))
    return feature, color, model_name, eval_method, params


def render_evaluation_tab():
    """Tab danh gia mo hinh linh hoat."""
    st.subheader("🧪 Danh gia mo hinh linh hoat")
    st.markdown(
        "Chon feature, khong gian mau, model va phuong phap danh gia. "
        "Ket qua se hien thi ngay tren UI va duoc luu vao `results/`."
    )

    feature, color, model_name, eval_method, params = render_evaluation_controls()

    if st.button("Chay thi nghiem", type="primary"):
        with st.spinner("Dang chay thi nghiem, vui long cho..."):
            result, report_text = run_experiment(
                feature=feature,
                color=color,
                model_name=model_name,
                eval_method=eval_method,
                test_size=params['test_size'],
                k=params['k'],
                n_repeats=params['n_repeats'],
                n_iterations=params['n_iterations'],
                sample_ratio=params['sample_ratio'],
                random_state=params['random_state'],
            )
        st.session_state['experiment_result'] = result
        st.session_state['experiment_report_text'] = report_text

    result = st.session_state.get('experiment_result')
    report_text = st.session_state.get('experiment_report_text')
    if not result or not report_text:
        st.info("Chua co ket qua thi nghiem. Hay chon cau hinh va bam 'Chay thi nghiem'.")
        return

    st.markdown("---")
    st.success(
        f"Da luu ket qua vao:\n- {result['artifacts']['json']}\n- {result['artifacts']['report']}\n- {result['artifacts']['confusion_matrix']}"
    )

    info_cols = st.columns(4)
    info_cols[0].write(f"**Feature:** {result['feature']}")
    info_cols[1].write(f"**Color:** {result['color_space'].upper()}")
    info_cols[2].write(f"**Model:** {result['model'].upper()}")
    info_cols[3].write(f"**Evaluation:** {result['evaluation_label']}")

    render_summary_metrics(result['summary'])

    st.write("**Thong tin du lieu**")
    st.write({
        'dataset_shape': result['dataset_shape'],
        'class_count': result['class_count'],
        'feature_file': result['feature_file'],
    })

    st.subheader("📋 Classification report")
    render_experiment_table(result['classification_report'])
    with st.expander("Xem classification report dang text"):
        st.text(report_text)

    st.subheader("🧩 Confusion matrix")
    st.image(result['artifacts']['confusion_matrix'])


def main():
    st.set_page_config(
        page_title="Color Correlogram - Nhan Dang Anh",
        page_icon="🎨",
        layout="wide",
    )

    st.title("🎨 Color Correlogram - Nhan Dang Anh")
    st.markdown(
        """
        **Do An Mon Khai Pha Du Lieu Da Phuong Tien**

        Ung dung nay gom 2 phan:
        1. **Nhan dang anh** bang model da train san.
        2. **Danh gia mo hinh linh hoat** theo feature / color space / model / evaluation method.
        """
    )

    tab_predict, tab_evaluate = st.tabs(["Nhan dang anh", "Danh gia mo hinh"])

    with tab_predict:
        render_prediction_tab()

    with tab_evaluate:
        render_evaluation_tab()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>Do An Mon Khai Pha Du Lieu Da Phuong Tien | Color Correlogram + ML Evaluation Toolkit</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
