"""
app.py - Ung dung demo Streamlit cho Color Correlogram Image Recognition

Chay: streamlit run app.py
"""

import os
import sys
import numpy as np
import cv2
import joblib
import streamlit as st
from pathlib import Path

# Them duong dan src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from color_correlogram import extract_correlogram_feature
from color_histogram import extract_histogram_feature


def load_model_and_data():
    """Tai model va du lieu da luu."""
    project_dir = Path(__file__).parent
    models_dir = project_dir / "models"
    features_dir = project_dir / "data" / "features"

    # Tai model
    model_path = models_dir / "svm_correlogram_hsv.pkl"
    if not model_path.exists():
        return None, None, None

    model = joblib.load(model_path)
    class_names = np.load(features_dir / "class_names.npy", allow_pickle=True)

    # Tai image paths de tim anh tuong tu
    paths_file = features_dir / "image_paths.npy"
    features_file = features_dir / "correlogram_hsv.npy"

    image_paths = None
    db_features = None
    if paths_file.exists() and features_file.exists():
        image_paths = np.load(paths_file, allow_pickle=True)
        db_features = np.load(features_file)

    return model, class_names, (image_paths, db_features)


def find_similar_images(query_feat, db_features, image_paths, top_k=3):
    """Tim cac anh tuong tu trong database.

    Args:
        query_feat: Vector dac trung cua anh query
        db_features: Ma tran dac trung cua database
        image_paths: Danh sach duong dan anh trong database
        top_k: So anh tra ve

    Returns:
        similar: List cac (path, distance)
    """
    from scipy.spatial.distance import cdist

    # Tinh khoang cach
    distances = cdist([query_feat], db_features, metric='cosine')[0]

    # Lay top-k gan nhat
    indices = np.argsort(distances)[:top_k]
    similar = [(image_paths[i], distances[i]) for i in indices]

    return similar


def main():
    st.set_page_config(
        page_title="Color Correlogram - Nhan Dang Anh",
        page_icon="🎨",
        layout="wide"
    )

    st.title("🎨 Color Correlogram - Nhan Dang Anh")
    st.markdown("""
    **Do An Mon Khai Pha Du Lieu Da Phuong Tien**

    Ung dung nay su dung phuong phap **Color Correlogram** de nhan dang anh.
    Upload 1 anh va he thong se du doan lop cua anh do.
    """)

    # Tai model
    model, class_names, db_data = load_model_and_data()

    if model is None:
        st.error("❌ Chua co model da train! Hay chay cac buoc sau truoc:")
        st.code("""
1. python src/feature_extraction.py   # Trich xuat dac trung
2. python src/train.py                # Huan luyen model
        """)
        return

    # Sidebar - Tham so
    st.sidebar.header("⚙️ Tham so")
    color_space = st.sidebar.selectbox("Khong gian mau:", ["HSV", "RGB"])
    h_bins = st.sidebar.slider("H bins:", 4, 16, 8)
    s_bins = st.sidebar.slider("S bins:", 2, 8, 3)
    v_bins = st.sidebar.slider("V bins:", 2, 8, 3)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Cac lop nhan dang:** {len(class_names)}")
    for i, name in enumerate(class_names):
        st.sidebar.write(f"  {i+1}. {name}")

    # Upload anh
    uploaded_file = st.file_uploader(
        "📁 Upload anh can nhan dang",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )

    if uploaded_file is not None:
        # Doc anh
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Anh goc")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=300)

        # Trich xuat dac trung
        with st.spinner("Dang trich xuat dac trung..."):
            feat = extract_correlogram_feature(
                img,
                color_space=color_space.lower(),
                h_bins=h_bins, s_bins=s_bins, v_bins=v_bins
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
            ax.set_title(f'Color Correlogram ({color_space})')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Du doan
        st.markdown("---")
        st.subheader("🎯 Ket qua du doan")

        with st.spinner("Dang du doan..."):
            # Model can feature co cung kich thuoc voi luc train
            # Neu color space khac hoac bins khac, feature size se khac
            try:
                prediction = model.predict([feat])
                predicted_class = class_names[prediction[0]]

                st.success(f"**Lop du doan: {predicted_class}**")

                # Hien thi anh tuong tu
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

            except Exception as e:
                st.error(f"Loi khi du doan: {e}")
                st.info("Co the do feature size khong khop voi model da train. "
                       "Hay dung cung tham so nhu khi train (HSV 8x3x3).")

    else:
        st.info("👆 Hay upload 1 anh de bat dau nhan dang!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>Do An Mon Khai Pha Du Lieu Da Phuong Tien | Color Correlogram + SVM</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
