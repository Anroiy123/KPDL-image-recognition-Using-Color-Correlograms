# Color Correlogram cho Nhận Dạng Ảnh

Đây là đồ án môn **Khai phá dữ liệu đa phương tiện**. Dự án xây dựng một pipeline nhận dạng ảnh dựa trên đặc trưng **Color Correlogram**, sau đó huấn luyện các mô hình học máy để phân lớp ảnh trong bộ dữ liệu **Corel-1K**.

Mục tiêu chính của repo:
- Tiền xử lý và chuẩn hóa ảnh đầu vào.
- Trích xuất đặc trưng **Color Correlogram** và **Color Histogram** để so sánh.
- Huấn luyện các mô hình **SVM**, **KNN** và **Random Forest**.
- Đánh giá mô hình bằng nhiều chỉ số và trực quan hóa kết quả.
- Cung cấp ứng dụng **Streamlit** để thử nhận dạng ảnh trực tiếp.

## 1. Tổng quan phương pháp

Color Correlogram không chỉ đếm tần suất màu như histogram mà còn mô tả **mối tương quan không gian giữa các màu** ở các khoảng cách xác định. Điều này giúp đặc trưng nắm bắt được cấu trúc hình ảnh tốt hơn trong bài toán nhận dạng lớp ảnh tự nhiên.

Pipeline hiện tại của dự án:
1. Đọc ảnh từ `data/corel-1k/` và resize về `128x128`.
2. Lượng tử hóa màu trong không gian **HSV** hoặc **RGB**.
3. Trích xuất đặc trưng:
   - `Correlogram HSV`: `72 x 4 = 288` chiều
   - `Spatial Correlogram HSV`: `(1 + 2x2) x 72 x 4 = 1440` chiều
   - `Correlogram RGB`: `256` chiều
   - `Histogram HSV`: `72` chiều
4. Huấn luyện các mô hình học máy với `GridSearchCV`.
5. Tạo split cố định `train/val/test` để tách riêng dữ liệu huấn luyện, chọn mô hình, và kiểm tra cuối.
6. Đánh giá bằng `Accuracy`, `Precision`, `Recall`, `F1-score`, `Classification Report`, `Confusion Matrix` trên tập `test` độc lập.
7. Chạy demo Streamlit để dự đoán lớp ảnh và tìm ảnh tương tự trong tập không dùng `test` để train model.

## 2. Cấu trúc thư mục

```text
KPDL/
├── data/
│   ├── corel-1k/                # Dataset Corel-1K, mỗi lớp là một thư mục con
│   ├── splits/                  # Metadata split train/val/test
│   └── features/                # Đặc trưng đã trích xuất (.npy)
├── models/                      # Model đã huấn luyện (.pkl)
├── notebooks/                   # Notebook minh họa và phân tích
├── results/                     # Báo cáo, biểu đồ, confusion matrix
├── src/
│   ├── preprocessing.py         # Đọc ảnh, resize, lượng tử hóa màu, load dataset
│   ├── color_correlogram.py     # Cài đặt Color Correlogram
│   ├── color_histogram.py       # Cài đặt Color Histogram
│   ├── feature_extraction.py    # Trích xuất đặc trưng cho toàn bộ dataset
│   ├── train.py                 # Huấn luyện SVM, KNN, Random Forest
│   └── evaluate.py              # Đánh giá mô hình và lưu biểu đồ
├── app.py                       # Ứng dụng Streamlit demo
├── requirements.txt             # Danh sách thư viện cần cài
└── README.md
```

## 3. Dữ liệu sử dụng

Dự án đang dùng bộ dữ liệu **Corel-1K** với:
- `10` lớp ảnh
- `100` ảnh mỗi lớp
- Tổng cộng `1000` ảnh

Các lớp hiện có trong repo:
- `africans`
- `beaches`
- `buildings`
- `buses`
- `dinosaurs`
- `elephants`
- `flowers`
- `food`
- `horses`
- `mountains`

Cấu trúc thư mục dữ liệu cần đúng như sau:

```text
data/corel-1k/
├── africans/
├── beaches/
├── buildings/
├── buses/
├── dinosaurs/
├── elephants/
├── flowers/
├── food/
├── horses/
└── mountains/
```

Mỗi thư mục lớp chứa các file ảnh như `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.

## 4. Yêu cầu môi trường

Cài các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Các thư viện chính đang được dùng:
- `numpy`
- `opencv-python`
- `Pillow`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter`
- `joblib`
- `tqdm`
- `streamlit`

## 5. Chạy nhanh

Nếu repo đã có sẵn:
- `data/features/*.npy`
- `models/*.pkl`

thì bạn có thể chạy demo ngay:

```bash
streamlit run app.py
```

Nếu muốn chạy lại toàn bộ pipeline từ đầu, làm theo các bước ở phần dưới.

## 6. Quy trình chạy đầy đủ

### Bước 1. Kiểm tra dữ liệu

Đảm bảo dataset nằm đúng tại:

```text
data/corel-1k/
```

Script `src/preprocessing.py` sẽ:
- Duyệt từng thư mục lớp theo thứ tự đã sắp xếp.
- Đọc ảnh bằng OpenCV.
- Resize toàn bộ ảnh về `128x128`.
- Hỗ trợ nhiều định dạng ảnh khác nhau.

### Bước 2. Trích xuất đặc trưng

Chạy:

```bash
python src/feature_extraction.py
```

Script này sẽ:
- Load toàn bộ dataset.
- Tạo 5 bộ đặc trưng:
  - `correlogram_hsv.npy`
  - `correlogram_hsv_spatial.npy`
  - `correlogram_rgb.npy`
  - `histogram_hsv.npy`
  - `histogram_rgb.npy`
- Đồng thời lưu:
  - `labels.npy`
  - `class_names.npy`
  - `image_paths.npy`
  - `data/splits/corel-1k_split.json`

Các file sinh ra trong `data/features/` và shape hiện tại trong repo là:

```text
correlogram_hsv.npy : (1000, 288)
correlogram_hsv_spatial.npy : (1000, 1440)
correlogram_rgb.npy : (1000, 256)
histogram_hsv.npy   : (1000, 72)
histogram_rgb.npy   : (1000, 64)
labels.npy          : (1000,)
class_names.npy     : (10,)
image_paths.npy     : (1000,)
```

Ý nghĩa:
- `correlogram_hsv_spatial.npy`: đặc trưng chính dùng cho mô hình demo tốt nhất.
- `correlogram_hsv.npy`: đặc trưng correlogram HSV toàn ảnh để so sánh.
- `correlogram_rgb.npy`: biến thể so sánh theo không gian màu RGB.
- `histogram_hsv.npy`: baseline để so sánh với Correlogram.
- `histogram_rgb.npy`: baseline histogram trong không gian RGB cho các thí nghiệm linh hoạt.
- `labels.npy`: nhãn lớp của từng ảnh.
- `class_names.npy`: danh sách tên lớp.
- `image_paths.npy`: đường dẫn ảnh gốc, dùng để tìm ảnh tương tự trong demo.
- `data/splits/corel-1k_split.json`: split cố định `train/val/test` dùng chung cho train, evaluation và app.

### Bước 3. Huấn luyện mô hình

Chạy:

```bash
python src/train.py
```

Script `src/train.py` hiện huấn luyện 6 thí nghiệm:
- `Spatial Correlogram HSV + SVM`
- `Correlogram HSV + SVM`
- `Correlogram HSV + KNN`
- `Correlogram HSV + Random Forest`
- `Histogram HSV + SVM`
- `Correlogram RGB + SVM`

Workflow của `src/train.py`:
- Tune hyperparameter chỉ trên `train`
- Đo tạm trên `val`
- Refit model cuối trên `train+val`
- Không dùng `test` trong huấn luyện

Model được lưu tại `models/`:
- `svm_correlogram_hsv_spatial.pkl`
- `svm_correlogram_hsv.pkl`
- `knn_correlogram_hsv.pkl`
- `rf_correlogram_hsv.pkl`
- `svm_histogram_hsv.pkl`
- `svm_correlogram_rgb.pkl`
- `*.meta.json`: metadata provenance, split file và split dùng để train model

Kết quả tổng hợp của quá trình train được lưu tại:

```text
results/training_results.json
```

### Bước 4. Đánh giá mô hình

Chạy:

```bash
python src/evaluate.py
```

Script `src/evaluate.py` sẽ:
- Load feature từ `data/features/`
- Load split metadata từ `data/splits/corel-1k_split.json`
- Load model từ `models/`
- Chỉ tạo dự đoán trên tập `test` độc lập
- Tính các chỉ số:
  - `Accuracy`
  - `Precision`
  - `Recall`
  - `F1-score`
- Lưu báo cáo văn bản và biểu đồ vào `results/`

Các file kết quả chính:
- `results/evaluation_summary.json`
- `results/accuracy_comparison.png`
- `results/per_class_comparison.png`
- `results/report_*.txt`
- `results/cm_*.png`

### Bước 5. Chạy ứng dụng demo Streamlit

Chạy:

```bash
streamlit run app.py
```

Ứng dụng `app.py` sẽ:
- Load model `models/svm_correlogram_hsv_spatial.pkl`
- Load metadata `models/svm_correlogram_hsv_spatial.meta.json`
- Load `class_names.npy`
- Load `image_paths.npy` và `correlogram_hsv_spatial.npy` để tìm ảnh tương tự trong split `train+val`
- Cho phép upload ảnh để dự đoán lớp
- Hiển thị vector đặc trưng Correlogram
- Hiển thị các ảnh gần nhất trong tập dữ liệu nếu có đủ dữ liệu phụ trợ

Lưu ý quan trọng:
- Model demo mặc định được huấn luyện với **HSV 8x3x3 + spatial grid 2x2**.
- Ứng dụng sẽ hiển thị thêm độ tin cậy dự đoán và provenance split của model.
- Ảnh ngoài tập 10 lớp Corel-1K có thể bị ép vào lớp gần nhất, ngay cả khi pipeline đánh giá đã chuẩn hơn.
- Nếu bạn thay đổi `color_space` hoặc các tham số `H bins`, `S bins`, `V bins` trên sidebar, vector đặc trưng có thể không khớp kích thước với model đã train.
- Khi đó ứng dụng có thể báo lỗi dự đoán và yêu cầu dùng lại tham số gốc.

## 7. Ý nghĩa của các thư mục output

### `data/features/`
Chứa đặc trưng đã trích xuất để tránh phải tính lại từ đầu mỗi lần huấn luyện.

### `data/splits/`
Chứa split metadata cố định để đảm bảo train, validation và final test luôn dùng đúng cùng một phép chia dữ liệu.

### `models/`
Chứa các model `.pkl` đã huấn luyện. Đây là đầu vào bắt buộc để `app.py` dự đoán được.

### `results/`
Chứa toàn bộ kết quả đánh giá và trực quan hóa:
- Báo cáo text cho từng mô hình
- Confusion matrix cho từng thí nghiệm
- Biểu đồ so sánh accuracy
- Biểu đồ so sánh accuracy theo từng lớp
- File JSON tổng hợp kết quả

## 8. Notebook trong dự án

Thư mục `notebooks/` gồm:
- `01_kham_pha_du_lieu.ipynb`: khám phá dữ liệu ban đầu
- `02_color_correlogram_demo.ipynb`: minh họa cách hoạt động của Color Correlogram
- `03_huan_luyen_danh_gia.ipynb`: notebook huấn luyện và đánh giá
- `04_so_sanh_ket_qua.ipynb`: tổng hợp và so sánh kết quả thực nghiệm

Chạy notebook:

```bash
cd notebooks
jupyter notebook
```

## 9. Kết quả hiện có trong repo

Kết quả được lưu trong `results/training_results.json` và `results/evaluation_summary.json` cho thấy:

| Phương pháp | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Spatial Correlogram HSV + SVM | 87.5% | 0.8740 | 0.8750 | 0.8735 |
| Correlogram HSV + SVM | 85.3% | 0.8560 | 0.8530 | 0.8528 |
| Correlogram HSV + KNN | 79.8% | 0.8192 | 0.7980 | 0.7926 |
| Correlogram HSV + Random Forest | 85.2% | 0.8515 | 0.8520 | 0.8498 |
| Histogram HSV + SVM | 79.5% | 0.7945 | 0.7950 | 0.7931 |
| Correlogram RGB + SVM | 82.9% | 0.8295 | 0.8290 | 0.8284 |

Nhận xét nhanh:
- **Spatial Correlogram HSV + SVM** là cấu hình tốt nhất trong các thí nghiệm hiện có.
- **Correlogram HSV** cho kết quả tốt hơn **Histogram HSV**, cho thấy thông tin tương quan không gian màu có ích cho bài toán này.
- Biến thể **RGB** vẫn hoạt động tốt, nhưng kém hơn cấu hình **HSV** tốt nhất.

## 10. Tóm tắt kỹ thuật

Các chi tiết chính trong phương pháp hiện tại:
- Không gian màu chính: `HSV`
- Lượng tử hóa HSV: `8 x 3 x 3 = 72` màu
- Khoảng cách correlogram: `{1, 3, 5, 7}`
- Vector đặc trưng chính: `1440` chiều (`Spatial Correlogram HSV`)
- Mô hình demo chính: `SVM`
- Tách dữ liệu: `train/val/test = 70/15/15`
- Phương pháp đánh giá cuối: `held-out test` trên split cố định

## 11. Lỗi thường gặp và cách xử lý

### Không tìm thấy dataset

Biểu hiện:
- Script báo không có ảnh hoặc không tìm thấy thư mục con trong `data/corel-1k/`

Cách xử lý:
- Kiểm tra lại đường dẫn `data/corel-1k/`
- Kiểm tra mỗi lớp có phải là một thư mục con riêng hay không
- Kiểm tra file ảnh có đuôi hợp lệ như `.jpg`, `.png`, `.bmp`, `.tif`

### Không chạy được `train.py`

Biểu hiện:
- Lỗi khi load `correlogram_hsv.npy`, `histogram_hsv.npy` hoặc `labels.npy`

Cách xử lý:
- Chạy lại:

```bash
python src/feature_extraction.py
```

- Sau đó mới chạy lại:

```bash
python src/train.py
```

### Không chạy được `evaluate.py`

Biểu hiện:
- Thiếu file model `.pkl`

Cách xử lý:
- Chạy lại bước huấn luyện:

```bash
python src/train.py
```

### Streamlit báo lỗi dự đoán

Biểu hiện:
- `Loi khi du doan`
- Ứng dụng gợi ý feature size không khớp với model

Cách xử lý:
- Dùng lại tham số mặc định khi demo:
  - `color_space = HSV`
  - `H bins = 8`
  - `S bins = 3`
  - `V bins = 3`
- Nếu chưa có model, chạy:

```bash
python src/feature_extraction.py
python src/train.py
```

### Không hiện ảnh tương tự trong demo

Biểu hiện:
- Không xuất hiện các ảnh gần nhất trong dataset

Cách xử lý:
- Kiểm tra `data/features/image_paths.npy`
- Kiểm tra `data/features/correlogram_hsv.npy`
- Đảm bảo đường dẫn ảnh gốc trong dataset vẫn còn tồn tại

## 12. Tham khảo

- Huang et al. (1997), *Image Indexing Using Color Correlograms*, CVPR.
