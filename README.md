# Color Correlogram cho Nhận Dạng Ảnh

Đây là đồ án môn **Khai phá dữ liệu đa phương tiện**. Dự án xây dựng một pipeline nhận dạng ảnh dựa trên đặc trưng **Color Correlogram**, sau đó huấn luyện các mô hình học máy để phân lớp ảnh trong bộ dữ liệu **Corel-1K**.

Mục tiêu chính của repo:
- Tiền xử lý và chuẩn hóa ảnh đầu vào.
- Trích xuất đặc trưng **Color Correlogram** và **Color Histogram** để so sánh.
- Huấn luyện các mô hình **SVM** và **KNN**.
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
5. Tạo split cố định `train/test` để tách riêng dữ liệu huấn luyện và kiểm tra cuối.
6. Đánh giá bằng `Accuracy`, `Precision`, `Recall`, `F1-score`, `Classification Report`, `Confusion Matrix` trên tập `test` độc lập.
7. Chạy demo Streamlit để dự đoán lớp ảnh và tìm ảnh tương tự trong tập không dùng `test` để train model.

## 2. Cấu trúc thư mục

```text
KPDL/
├── data/
│   ├── corel-1k/                # Dataset Corel-1K, mỗi lớp là một thư mục con
│   ├── splits/                  # Metadata split train/test
│   └── features/                # Đặc trưng đã trích xuất (.npy)
├── models/                      # Model đã huấn luyện (.pkl)
├── notebooks/                   # Notebook minh họa và phân tích
├── results/                     # Báo cáo, biểu đồ, confusion matrix
├── src/
│   ├── preprocessing.py         # Đọc ảnh, resize, lượng tử hóa màu, load dataset
│   ├── color_correlogram.py     # Cài đặt Color Correlogram
│   ├── color_histogram.py       # Cài đặt Color Histogram
│   ├── feature_extraction.py    # Trích xuất đặc trưng cho toàn bộ dataset
│   ├── train.py                 # Huấn luyện SVM và KNN
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
- `data/splits/corel-1k_split.json`: split cố định `train/test` dùng chung cho train, evaluation và app.

### Bước 3. Huấn luyện mô hình

Chạy:

```bash
python src/train.py
```

Script `src/train.py` hiện huấn luyện 5 thí nghiệm:
- `Spatial Correlogram HSV + SVM`
- `Correlogram HSV + SVM`
- `Correlogram HSV + KNN`
- `Histogram HSV + SVM`
- `Correlogram RGB + SVM`

Workflow của `src/train.py`:
- Tune hyperparameter bằng `StratifiedKFold` chỉ trên `train`
- Lưu model cuối trên toàn bộ `train`
- Không dùng `test` trong huấn luyện

Model được lưu tại `models/`:
- `svm_correlogram_hsv_spatial.pkl`
- `svm_correlogram_hsv.pkl`
- `knn_correlogram_hsv.pkl`
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
- Load `image_paths.npy` và `correlogram_hsv_spatial.npy` để tìm ảnh tương tự trong split `train`
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
Chứa split metadata cố định để đảm bảo train và final test luôn dùng đúng cùng một phép chia dữ liệu.

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
| Spatial Correlogram HSV + SVM | 84.5% | 0.8462 | 0.8450 | 0.8442 |
| Correlogram HSV + SVM | 82.5% | 0.8313 | 0.8250 | 0.8259 |
| Correlogram HSV + KNN | 78.0% | 0.8233 | 0.7800 | 0.7790 |
| Histogram HSV + SVM | 78.0% | 0.7798 | 0.7800 | 0.7778 |
| Correlogram RGB + SVM | 82.0% | 0.8294 | 0.8200 | 0.8188 |

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
- Tách dữ liệu: `train/test = 80/20`
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

## 13. Kết Quả Thực Nghiệm

Kết quả được lưu trong `results/training_results.json` và `results/evaluation_summary.json` cho thấy:

| Phương pháp | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Spatial Correlogram HSV + SVM | 84.5% | 0.8462 | 0.8450 | 0.8442 |
| Correlogram HSV + SVM | 82.5% | 0.8313 | 0.8250 | 0.8259 |
| Correlogram HSV + KNN | 78.0% | 0.8233 | 0.7800 | 0.7790 |
| Histogram HSV + SVM | 78.0% | 0.7798 | 0.7800 | 0.7778 |
| Correlogram RGB + SVM | 82.0% | 0.8294 | 0.8200 | 0.8188 |

Nhận xét nhanh:
- **Spatial Correlogram HSV + SVM** là cấu hình tốt nhất trong các thí nghiệm hiện có.
- **Correlogram HSV** cho kết quả tốt hơn **Histogram HSV**, cho thấy thông tin tương quan không gian màu có ích cho bài toán này.
- Biến thể **RGB** vẫn hoạt động tốt, nhưng kém hơn cấu hình **HSV** tốt nhất.

## 14. Hạn Chế

Dự án hiện tại có các hạn chế sau:

- **Kích thước dataset**: Chỉ sử dụng 1000 ảnh từ 10 lớp Corel-1K. Các dataset lớn hơn (ImageNet, COCO) có thể cải thiện độ tổng quát của mô hình.
- **Không gian màu**: Chỉ thử nghiệm HSV và RGB. Các không gian màu khác như LAB, YCbCr có thể cung cấp thông tin bổ sung.
- **Số lượng lớp**: Giới hạn ở 10 lớp. Các bài toán phân lớp với số lượng lớp lớn hơn (100+) có thể yêu cầu kiến trúc mô hình khác.
- **Độ phức tạp tính toán**: Color Correlogram có độ phức tạp O(H*W*d*8) với d là số khoảng cách. Các dataset lớn hơn hoặc ảnh độ phân giải cao có thể gặp vấn đề hiệu năng.
- **Mô hình học máy**: Chỉ sử dụng SVM và KNN. Các mô hình học sâu (CNN, ResNet) có thể đạt độ chính xác cao hơn.

## 15. Hướng Phát Triển Tương Lai

Các hướng cải thiện và mở rộng cho dự án:

- **Dataset lớn hơn**: Sử dụng ImageNet, COCO, hoặc các dataset công khai khác để huấn luyện mô hình với khả năng tổng quát hóa tốt hơn.
- **Học sâu**: Triển khai CNN (VGG, ResNet, EfficientNet) để trích xuất đặc trưng tự động và đạt độ chính xác cao hơn.
- **Xử lý thời gian thực**: Tối ưu hóa pipeline để xử lý video hoặc stream ảnh từ camera với độ trễ thấp.
- **Mở rộng không gian màu**: Thử nghiệm LAB, YCbCr, HSL và các không gian màu khác để tìm ra không gian tối ưu cho bài toán.
- **Kết hợp đặc trưng**: Kết hợp Color Correlogram với các đặc trưng khác (SIFT, ORB, Texture) để cải thiện hiệu suất.
- **Tìm kiếm ảnh tương tự**: Xây dựng hệ thống tìm kiếm ảnh dựa trên độ tương tự để ứng dụng trong e-commerce, thư viện ảnh.
- **Triển khai trên thiết bị di động**: Chuyển đổi mô hình sang TensorFlow Lite hoặc ONNX để chạy trên điện thoại thông minh.

## 16. Thesis Document Generator

### Tổng Quan

Thesis Document Generator là một skill Kiro cho phép tạo các tài liệu tiêu luận học thuật định dạng chuẩn Vietnamese dưới dạng DOCX. Skill này tích hợp với thư mục `results/` để tự động đưa các hình ảnh, biểu đồ và ma trận nhầm lẫn vào tài liệu.

### Cài Đặt

Thư viện `python-docx` đã được thêm vào `requirements.txt`. Cài đặt:

```bash
pip install -r requirements.txt
```

### Cấu Trúc Module

Module `src/thesis_generator.py` cung cấp ba lớp chính:

1. **ThesisDocument**: Lớp chính quản lý tài liệu, định dạng, và lưu file DOCX
2. **ThesisChapter**: Đại diện cho một chương với nội dung, hình ảnh, và bảng
3. **AssetManager**: Quản lý tải hình ảnh từ thư mục `results/`

### Sử Dụng Cơ Bản

#### Tạo Tài Liệu Từ Config Dict

```python
from src.thesis_generator import ThesisDocument

config = {
    "title": "Tiêu Luận Khái Phá Dữ Liệu",
    "chapters": [
        {
            "number": 1,
            "title": "Tổng quan",
            "content": [
                {"type": "paragraph", "text": "Đây là nội dung chương 1..."},
                {"type": "image", "path": "results/cm_model.png", "caption": "Ma trận nhầm lẫn"},
                {"type": "table", "data": [["Cột 1", "Cột 2"], ["Dữ liệu 1", "Dữ liệu 2"]], "caption": "Bảng kết quả"}
            ]
        },
        {
            "number": 2,
            "title": "Cơ sở lý thuyết",
            "content": [
                {"type": "paragraph", "text": "Nội dung chương 2..."}
            ]
        }
    ]
}

doc = ThesisDocument()
doc.generate_from_config(config)
doc.save("my_thesis.docx")
```

#### Tạo Tài Liệu Từng Chương

```python
from src.thesis_generator import ThesisDocument, ThesisChapter

doc = ThesisDocument("Tiêu Luận Khái Phá Dữ Liệu")

# Chương 1
chapter1 = ThesisChapter(1, "Tổng quan")
chapter1.add_paragraph("Đây là nội dung chương 1...")
chapter1.add_image("results/cm_model.png", "Ma trận nhầm lẫn")
doc.add_chapter(chapter1)

# Chương 2
chapter2 = ThesisChapter(2, "Cơ sở lý thuyết")
chapter2.add_paragraph("Nội dung chương 2...")
doc.add_chapter(chapter2)

doc.save("my_thesis.docx")
```

#### Sử Dụng Skill Từ CLI

```bash
python skills/generate_thesis_doc.py
```

### Định Dạng Hỗ Trợ

- **Font**: Times New Roman (12pt)
- **Căn lề**: Justified (căn đều hai bên)
- **Khoảng cách dòng**: 1.5
- **Hình ảnh**: Tự động đánh số "Hình X" với chú thích dưới hình, căn giữa
- **Bảng**: Tự động đánh số "Bảng X" với chú thích trên bảng, căn trái
- **Hỗ trợ UTF-8**: Đầy đủ hỗ trợ tiếng Việt

### Quản Lý Tài Sản (Asset Manager)

AssetManager tự động quét thư mục `results/` để tìm hình ảnh:

```python
from src.thesis_generator import AssetManager

manager = AssetManager("results/")

# Liệt kê tất cả hình ảnh
images = manager.list_available_results()
print(images)

# Tìm ma trận nhầm lẫn
matrices = manager.find_confusion_matrices()
print(matrices)

# Tìm biểu đồ so sánh độ chính xác
charts = manager.find_accuracy_charts()
print(charts)

# Tải hình ảnh từ results/
path = manager.load_result_image("cm_model.png")
```

### Cấu Trúc Config Chi Tiết

```python
config = {
    "title": "Tiêu Luận Khái Phá Dữ Liệu",  # Tiêu đề tài liệu
    "chapters": [
        {
            "number": 1,                      # Số chương
            "title": "Tổng quan",             # Tiêu đề chương
            "content": [                      # Nội dung chương
                {
                    "type": "paragraph",
                    "text": "Nội dung đoạn văn..."
                },
                {
                    "type": "image",
                    "path": "results/image.png",
                    "caption": "Chú thích hình ảnh",
                    "width": 5.0                # Chiều rộng inch (tùy chọn, mặc định 5.0)
                },
                {
                    "type": "table",
                    "data": [
                        ["Cột 1", "Cột 2", "Cột 3"],
                        ["Dữ liệu 1", "Dữ liệu 2", "Dữ liệu 3"],
                        ["Dữ liệu 4", "Dữ liệu 5", "Dữ liệu 6"]
                    ],
                    "caption": "Chú thích bảng"
                }
            ]
        }
    ]
}
```

### Ví Dụ Tạo Tiêu Luận 5 Chương

```python
from src.thesis_generator import ThesisDocument

config = {
    "title": "Tiêu Luận Khái Phá Dữ Liệu",
    "chapters": [
        {
            "number": 1,
            "title": "Tổng quan",
            "content": [
                {"type": "paragraph", "text": "Chương 1 trình bày tổng quan về khái phá dữ liệu..."}
            ]
        },
        {
            "number": 2,
            "title": "Cơ sở lý thuyết",
            "content": [
                {"type": "paragraph", "text": "Chương 2 trình bày các khái niệm cơ bản..."}
            ]
        },
        {
            "number": 3,
            "title": "Thiết kế và xây dựng hệ thống",
            "content": [
                {"type": "paragraph", "text": "Chương 3 trình bày thiết kế hệ thống..."}
            ]
        },
        {
            "number": 4,
            "title": "Cài đặt",
            "content": [
                {"type": "paragraph", "text": "Chương 4 trình bày chi tiết cài đặt..."}
            ]
        },
        {
            "number": 5,
            "title": "Kết luận",
            "content": [
                {"type": "paragraph", "text": "Chương 5 trình bày kết luận và hướng phát triển..."}
            ]
        }
    ]
}

doc = ThesisDocument()
doc.generate_from_config(config)
doc.save("thesis_output.docx")
```

### Xử Lý Lỗi Thường Gặp

#### Không tìm thấy hình ảnh

```
FileNotFoundError: Image file not found: results/image.png
```

**Giải pháp**: Kiểm tra đường dẫn hình ảnh tồn tại trong thư mục `results/`

#### Dữ liệu bảng không hợp lệ

```
ValueError: Table data must be a list of rows (each row is a list of cells)
```

**Giải pháp**: Đảm bảo dữ liệu bảng là danh sách các hàng, mỗi hàng là danh sách các ô

#### Hình ảnh không hiển thị trong Word

**Giải pháp**: Đảm bảo định dạng hình ảnh là `.png`, `.jpg`, hoặc `.jpeg`

### Kiểm Tra Chất Lượng

Chạy unit tests:

```bash
pytest tests/test_thesis_generator.py -v
```

Các test bao gồm:
- Tạo chương và thêm nội dung
- Quản lý tài sản từ thư mục results/
- Tạo tài liệu từ config dict
- Hỗ trợ UTF-8 tiếng Việt
- Lưu và mở file DOCX

## 17. Hướng Dẫn Tái Tạo Kết Quả

Để tái tạo lại toàn bộ kết quả từ đầu, làm theo các bước sau:

### Bước 1: Chuẩn Bị Môi Trường

```bash
# Clone repository
git clone <repository-url>
cd KPDL

# Tạo virtual environment (tùy chọn)
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Chuẩn Bị Dữ Liệu

Đảm bảo dataset Corel-1K nằm tại `data/corel-1k/` với cấu trúc:

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

### Bước 3: Trích Xuất Đặc Trưng

```bash
python src/feature_extraction.py
```

Lệnh này sẽ tạo ra các file `.npy` trong `data/features/` và split metadata trong `data/splits/`.

### Bước 4: Huấn Luyện Mô Hình

```bash
python src/train.py
```

Lệnh này sẽ huấn luyện 5 mô hình và lưu vào `models/` cùng với metadata `.meta.json`.

### Bước 5: Đánh Giá Mô Hình

```bash
python src/evaluate.py
```

Lệnh này sẽ đánh giá các mô hình trên tập test độc lập và lưu kết quả vào `results/`.

### Bước 6: Chạy Ứng Dụng Demo

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`.

### Bước 7: Chạy Kiểm Tra Chất Lượng Mã

```bash
# Chạy unit tests
pytest tests/ --cov=src --cov-report=html

# Kiểm tra type hints
mypy --strict src/

# Kiểm tra linting
pylint src/

# Format code
black src/ app.py
```

### Bước 8: Xác Minh Tái Tạo

Để xác minh rằng kết quả được tái tạo chính xác:

```bash
# Chạy pipeline hai lần với cùng seed
python src/feature_extraction.py
python src/train.py
python src/evaluate.py

# Lưu kết quả lần 1
cp results/evaluation_summary.json results/eval_run1.json

# Chạy lại
python src/train.py
python src/evaluate.py

# So sánh kết quả
diff results/eval_run1.json results/evaluation_summary.json
```

Nếu hai lần chạy cho kết quả giống nhau, pipeline là tái tạo được.
