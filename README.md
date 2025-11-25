# TB Detection Project - Phát hiện Lao phổi bằng YOLO

## 📋 Tổng quan

Project này sử dụng YOLOv8/YOLO11 để phát hiện bệnh lao phổi (Tuberculosis - TB) từ ảnh X-quang. Project được xây dựng trên framework Ultralytics YOLO với pipeline hoàn chỉnh từ preprocessing, training, evaluation đến visualization.

## 🎯 Mục đích

- **Phát hiện bệnh lao phổi**: Phân loại và phát hiện vùng bệnh lao trong ảnh X-quang
- **3 lớp phân loại**:
  - `healthy`: Khỏe mạnh
  - `sick_but_no_tb`: Bệnh nhưng không phải lao
  - `tb`: Bệnh lao phổi
- **Object Detection**: Phát hiện và vẽ bounding box cho các vùng bệnh

## 📁 Cấu trúc Project

```
ai4life/
├── main.py                      # Script chính để chạy toàn bộ pipeline
├── preprocessing.py             # Xử lý và chuyển đổi dữ liệu
├── train.py                     # Huấn luyện model
├── evaluate.py                  # Đánh giá model
├── heatmap.py                   # Tạo heatmap visualization
├── predict_with_json.py        # Dự đoán và trả về JSON
├── analyze_data.py              # Phân tích dữ liệu (chỉ phân tích)
├── data.py                      # Đếm số lượng ảnh
├── data_check.py                # Kiểm tra dữ liệu
├── test-model.py                # Test model nhanh
├── config_yolo.py               # Tạo config YOLO
├── convert_to_tensorboard.py    # Chuyển đổi kết quả sang TensorBoard
├── requirements.txt             # Dependencies
├── results.csv                   # Kết quả training (metrics)
├── best.pt                       # Model tốt nhất
├── tbx11k-simplified/           # Dataset đã xử lý
│   ├── data.csv                 # Metadata dataset
│   ├── dataset.yaml             # Config YOLO
│   ├── images/                  # Ảnh đã chia train/val
│   └── labels/                  # Labels YOLO format
├── yolo-dataset/                # Dataset YOLO format
├── results/                      # Thư mục lưu kết quả
│   ├── analysis/                # Kết quả phân tích
│   ├── evaluation/              # Kết quả đánh giá
│   ├── heatmaps/                # Heatmaps
│   └── metrics/                 # Metrics
└── tb_detection/                # Kết quả training YOLO
```

## 🔄 Luồng hoạt động

### Pipeline chính

```
1. PREPROCESSING (preprocessing.py)
   ├── Đọc CSV metadata
   ├── Phân tích dataset (analyze_dataset)
   ├── Chuyển đổi sang YOLO format
   └── Tạo dataset.yaml

2. TRAINING (train.py)
   ├── Load YOLOv8 model
   ├── Huấn luyện với dataset
   ├── Validation tự động
   └── Lưu best.pt và last.pt

3. EVALUATION (evaluate.py)
   ├── Đánh giá trên validation set
   ├── Tính metrics (mAP, Precision, Recall)
   └── Dự đoán trên ảnh test

4. VISUALIZATION (heatmap.py)
   ├── Tạo heatmap từ predictions
   ├── Vẽ bounding boxes
   └── Overlay lên ảnh gốc

5. PREDICTION (predict_with_json.py)
   ├── Dự đoán trên ảnh
   ├── Trả về JSON với bbox, confidence
   └── Lưu ảnh đã đánh dấu
```

## 📄 Chi tiết từng file

### 🚀 File chính

#### `main.py`

**Mục đích**: Entry point chính của project, điều phối toàn bộ pipeline

**Chức năng**:

- Parse arguments từ command line
- Chạy các bước: preprocess → train → evaluate → heatmap
- Hỗ trợ mode: `preprocess`, `train`, `evaluate`, `heatmap`, `all`

**Cách dùng**:

```bash
python main.py --mode all --csv data.csv --images images/ --output output/
```

#### `preprocessing.py`

**Mục đích**: Xử lý và chuyển đổi dữ liệu từ CSV sang YOLO format

**Class**: `TBDataPreprocessor`

**Chức năng chính**:

- `parse_csv()`: Đọc và parse file CSV metadata
- `analyze_dataset()`: Phân tích thống kê dataset
  - Phân bố theo source (train/val)
  - Phân bố theo target (no_tb/tb)
  - Phân bố theo image_type
  - Đếm số ảnh có bounding box
- `convert_to_yolo_format()`: Chuyển đổi bbox sang YOLO format (normalized)
- `create_yaml_config()`: Tạo file dataset.yaml cho YOLO
- `run()`: Chạy toàn bộ pipeline preprocessing

**Input**: CSV file với columns: `fname`, `image_height`, `image_width`, `source`, `bbox`, `target`, `image_type`

**Output**:

- Thư mục YOLO format (images/train, images/val, labels/train, labels/val)
- File `dataset.yaml`

#### `train.py`

**Mục đích**: Huấn luyện YOLOv8 model

**Class**: `TBModelTrainer`

**Chức năng**:

- Load YOLOv8 model (nano, small, medium, large, xlarge)
- Training với các hyperparameters:
  - Optimizer: SGD
  - Learning rate: 0.01
  - Data augmentation (mosaic, flip, HSV, etc.)
  - Early stopping (patience=20)
- Tự động validation sau mỗi epoch
- Lưu checkpoint mỗi 10 epochs
- Tạo plots và metrics

**Output**:

- `best.pt`: Model tốt nhất
- `last.pt`: Model cuối cùng
- Kết quả trong `tb_detection/`

#### `evaluate.py`

**Mục đích**: Đánh giá model trên validation set

**Class**: `TBModelEvaluator`

**Chức năng**:

- `evaluate_on_validation()`: Tính metrics (mAP50, mAP50-95, Precision, Recall)
- `predict_single_image()`: Dự đoán trên 1 ảnh
- `predict_batch()`: Dự đoán trên nhiều ảnh

**Metrics**:

- mAP50: Mean Average Precision @ IoU=0.5
- mAP50-95: Mean Average Precision @ IoU=0.5:0.95
- Precision: Độ chính xác
- Recall: Độ nhạy

#### `heatmap.py`

**Mục đích**: Tạo heatmap visualization từ predictions

**Class**: `TBHeatmapGenerator`

**Chức năng**:

- `generate_heatmap()`: Tạo heatmap cho 1 ảnh
  - Predict bounding boxes
  - Tạo gradient heatmap từ tâm bbox
  - Overlay lên ảnh gốc
  - Vẽ bounding boxes và labels
- `generate_batch_heatmaps()`: Tạo heatmap cho nhiều ảnh

**Visualization**:

- Heatmap màu JET (xanh → đỏ)
- Bounding boxes với màu theo class:
  - Green: healthy
  - Orange: sick_but_no_tb
  - Red: tb

#### `predict_with_json.py`

**Mục đích**: API để dự đoán và trả về kết quả dưới dạng JSON

**Class**: `TBDetectionAPI`

**Chức năng**:

- `predict_image()`: Dự đoán 1 ảnh, trả về JSON
  - Bounding boxes
  - Confidence scores
  - Class names
  - Ảnh đã đánh dấu (base64)
- `predict_batch()`: Dự đoán nhiều ảnh, lưu vào JSON file

**JSON format**:

```json
{
  "model_version": "yolov8n-tb-v1.0",
  "image_name": "test.png",
  "image_size": {"width": 512, "height": 512},
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class_id": 2,
      "class_name": "tb"
    }
  ],
  "annotated_image": "data:image/png;base64,..."
}
```

### 🔧 File hỗ trợ

#### `analyze_data.py`

**Mục đích**: Script đơn giản để chỉ phân tích dữ liệu (không chuyển đổi)

**Cách dùng**:

```bash
python analyze_data.py --csv data.csv --images images/
```

#### `data.py`

**Mục đích**: Đếm số lượng ảnh trong các thư mục train/val/test

#### `data_check.py`

**Mục đích**: Kiểm tra dữ liệu

- Kiểm tra ảnh và label có khớp không
- Phát hiện ảnh thiếu label hoặc label thiếu ảnh
- Kiểm tra encoding của file label

#### `test-model.py`

**Mục đích**: Script nhanh để test model

**Cách dùng**:

```python
python test-model.py
```

#### `config_yolo.py`

**Mục đích**: Tạo file dataset.yaml cho YOLO

#### `convert_to_tensorboard.py`

**Mục đích**: Chuyển đổi kết quả training từ CSV sang TensorBoard logs

**Cách dùng**:

```bash
python convert_to_tensorboard.py
# Sau đó: tensorboard --logdir runs/experiment_1
```

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy toàn bộ pipeline

```bash
# Chạy tất cả: preprocess → train → evaluate
python main.py --mode all \
    --csv tbx11k-simplified/data.csv \
    --images tbx11k-simplified/images \
    --output tbx11k-simplified \
    --epochs 100 \
    --batch 16 \
    --img-size 512
```

### 3. Chạy từng bước riêng lẻ

#### Bước 1: Preprocessing

```bash
python main.py --mode preprocess \
    --csv tbx11k-simplified/data.csv \
    --images tbx11k-simplified/images \
    --output tbx11k-simplified
```

#### Bước 2: Training

```bash
python main.py --mode train \
    --output tbx11k-simplified \
    --epochs 100 \
    --batch 16 \
    --img-size 512
```

#### Bước 3: Evaluation

```bash
python main.py --mode evaluate \
    --model tb_detection/yolov8n_tb_20251011_190341/weights/best.pt \
    --output tbx11k-simplified
```

#### Bước 4: Tạo heatmap

```bash
python main.py --mode heatmap \
    --model best.pt
```

### 4. Chỉ phân tích dữ liệu

```bash
python analyze_data.py --csv tbx11k-simplified/data.csv
```

### 5. Dự đoán với JSON API

```python
from predict_with_json import TBDetectionAPI

detector = TBDetectionAPI(model_path="best.pt")
result = detector.predict_image("test_image.png")
print(result)
```

## 📊 Dataset

### Cấu trúc CSV

File `data.csv` cần có các columns:

- `fname`: Tên file ảnh
- `image_height`, `image_width`: Kích thước ảnh
- `source`: `train` hoặc `val`
- `bbox`: Bounding box dạng dict `{'xmin': x, 'ymin': y, 'width': w, 'height': h}` hoặc `'none'`
- `target`: `no_tb` hoặc `tb`
- `image_type`: `healthy`, `sick_but_no_tb`, hoặc `tb`
- `tb_type`: (tùy chọn) Loại TB

### Classes

- **Class 0**: `healthy` - Khỏe mạnh
- **Class 1**: `sick_but_no_tb` - Bệnh nhưng không phải lao
- **Class 2**: `tb` - Bệnh lao phổi

## 📈 Metrics

Khi training, model sẽ tính các metrics:

- **mAP50**: Mean Average Precision @ IoU=0.5
- **mAP50-95**: Mean Average Precision @ IoU=0.5:0.95
- **Precision**: Độ chính xác
- **Recall**: Độ nhạy

Kết quả được lưu trong:

- `results.csv`: Metrics theo epoch
- `tb_detection/[run_name]/`: Plots, confusion matrix, PR curves

## 🎨 Visualization

### Heatmap

Tạo heatmap để visualize vùng phát hiện bệnh:

- Màu xanh → đỏ: Intensity tăng dần
- Bounding boxes với màu theo class
- Confidence scores

### TensorBoard

Xem training progress:

```bash
tensorboard --logdir tb_detection/
```

## 🔍 Troubleshooting

### Lỗi: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### Lỗi: Không tìm thấy ảnh

- Kiểm tra đường dẫn trong CSV
- Đảm bảo ảnh tồn tại trong thư mục `images/`

### Lỗi: Encoding label

- Chạy `data_check.py` để kiểm tra
- Đảm bảo file label là UTF-8

## 📝 Notes

- Model được lưu tự động trong `tb_detection/[timestamp]/weights/`
- Best model: `best.pt`
- Last model: `last.pt`
- Checkpoints mỗi 10 epochs: `epochN.pt`

## 🤝 Đóng góp

Project này được xây dựng cho mục đích nghiên cứu và giáo dục về phát hiện bệnh lao phổi bằng deep learning.

## 📄 License

[Thêm license nếu có]

---

**Tác giả**: [Tên tác giả]  
**Ngày tạo**: 2024  
**Phiên bản**: 1.0
