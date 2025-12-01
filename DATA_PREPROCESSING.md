# 📊 Tổng Quan Tiền Xử Lý Dữ Liệu

> Tài liệu chi tiết về quy trình tiền xử lý dữ liệu X-quang ngực cho hệ thống phát hiện Lao phổi (TB) sử dụng YOLOv8.

## 📋 Mục Lục

1. [Tổng Quan Dataset Gốc](#1-tổng-quan-dataset-gốc)
2. [Kiến Trúc Pipeline](#2-kiến-trúc-pipeline)
3. [Stage 1: Classification Dataset](#3-stage-1-classification-dataset)
4. [Stage 2: Detection Dataset](#4-stage-2-detection-dataset)
5. [Kỹ Thuật Augmentation](#5-kỹ-thuật-augmentation)
6. [Cấu Trúc Thư Mục Output](#6-cấu-trúc-thư-mục-output)
7. [Hướng Dẫn Sử Dụng](#7-hướng-dẫn-sử-dụng)

---

## 1. Tổng Quan Dataset Gốc

### 1.1 Nguồn Dữ Liệu: TBX11K-Simplified

| Thông số | Giá trị |
|----------|---------|
| **Tổng số ảnh** | 8,812 ảnh |
| **Định dạng** | PNG |
| **Kích thước** | 512 × 512 pixels |
| **File metadata** | `data.csv` |

### 1.2 Cấu Trúc File CSV

```csv
fname,image_height,image_width,source,bbox,target,tb_type,image_type
h0001.png,512,512,train,none,no_tb,none,healthy
tb_0001.png,512,512,train,"{'xmin': 180, 'ymin': 120, 'width': 150, 'height': 200}",tb,active_tb,active_tb
```

### 1.3 Các Trường Dữ Liệu

| Trường | Mô tả | Giá trị |
|--------|-------|---------|
| `fname` | Tên file ảnh | `h0001.png`, `tb_0001.png`, ... |
| `image_height` | Chiều cao ảnh | 512 |
| `image_width` | Chiều rộng ảnh | 512 |
| `source` | Nguồn/split | `train`, `val` |
| `bbox` | Bounding box | `none` hoặc `{'xmin':..., 'ymin':..., 'width':..., 'height':...}` |
| `target` | Nhãn mục tiêu | `no_tb`, `tb` |
| `tb_type` | Loại TB | `none`, `active_tb`, `latent_tb` |
| `image_type` | Loại ảnh | `healthy`, `sick_but_no_tb`, `active_tb`, `latent_tb` |

### 1.4 Phân Bố Dữ Liệu Gốc

```
📊 PHÂN BỐ 4 LỚP:
┌─────────────────┬──────────┬──────────┐
│ Lớp             │ Số lượng │ Tỷ lệ    │
├─────────────────┼──────────┼──────────┤
│ healthy         │ 3,800    │ 43.1%    │
│ sick_but_no_tb  │ 3,600    │ 40.9%    │
│ active_tb       │ 924      │ 10.5%    │
│ latent_tb       │ 239      │ 2.7%     │ ⚠️ Thiểu số
├─────────────────┼──────────┼──────────┤
│ TỔNG            │ 8,812    │ 100%     │
└─────────────────┴──────────┴──────────┘

📊 SỐ ẢNH CÓ BOUNDING BOX:
   → active_tb: ~924 ảnh có bbox
   → latent_tb: ~239 ảnh có bbox
   → TỔNG: ~1,163 ảnh có annotation
```

---

## 2. Kiến Trúc Pipeline

### 2.1 Tổng Quan Quy Trình

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE TIỀN XỬ LÝ DỮ LIỆU                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  📁 RAW DATA                                                                │
│  ┌─────────────────┐     ┌─────────────────┐                               │
│  │  data.csv       │     │  images/        │                               │
│  │  (metadata)     │     │  (8,812 ảnh)    │                               │
│  └────────┬────────┘     └────────┬────────┘                               │
└───────────┼───────────────────────┼─────────────────────────────────────────┘
            │                       │
            └───────────┬───────────┘
                        ▼
          ┌─────────────────────────────┐
          │  🔄 DataPreparer.load_data()│
          │  - Đọc CSV                  │
          │  - Phân loại 4 class        │
          │  - Phân tích thống kê       │
          └─────────────┬───────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │  🔄 Offline Augmentation    │
          │  - Tăng cường latent_tb     │
          │  - 239 → 640 ảnh            │
          └─────────────┬───────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
┌───────────────────────┐ ┌───────────────────────┐
│  📁 CLASSIFICATION    │ │  📁 DETECTION         │
│  (dataset_cls)        │ │  (dataset_det)        │
│                       │ │                       │
│  • 4 lớp              │ │  • 2 lớp + background │
│  • Tất cả 8,812+ ảnh  │ │  • 1,163+ ảnh TB      │
│  • Chia 80/20         │ │  • 800+ ảnh background│
│  • Cấu trúc folder    │ │  • YOLO format bbox   │
└───────────────────────┘ └───────────────────────┘
```

### 2.2 Hai Luồng Xử Lý Song Song

| Aspect | Classification | Detection |
|--------|---------------|-----------|
| **Mục đích** | Phân loại 4 loại X-quang | Định vị tổn thương TB |
| **Số lớp** | 4 | 2 |
| **Dữ liệu** | Tất cả ảnh | Chỉ ảnh có bbox + background |
| **Format** | Folder-based | YOLO format (txt) |

---

## 3. Stage 1: Classification Dataset

### 3.1 Mục Tiêu

Tạo dataset cho **YOLOv8-cls** với nhiệm vụ phân loại ảnh X-quang vào 4 lớp:

```
┌─────────────────────────────────────────────────────────────┐
│                  4 LỚP PHÂN LOẠI                           │
├─────────────────┬──────────────────────────────────────────┤
│ Class 0         │ healthy        - Khỏe mạnh               │
│ Class 1         │ sick_but_no_tb - Bệnh nhưng không phải TB│
│ Class 2         │ active_tb      - Lao hoạt động           │
│ Class 3         │ latent_tb      - Lao tiềm ẩn             │
└─────────────────┴──────────────────────────────────────────┘
```

### 3.2 Logic Xác Định Lớp

```python
def _get_class_name(row):
    if row['image_type'] == 'healthy':
        return 'healthy'
    elif row['image_type'] == 'sick_but_no_tb':
        return 'sick_but_no_tb'
    elif row['target'] == 'tb':
        if row['tb_type'] == 'latent_tb':
            return 'latent_tb'
        else:
            return 'active_tb'  # Mặc định cho TB không rõ loại
    return 'healthy'  # Fallback
```

### 3.3 Cấu Trúc Output

```
dataset_cls/
├── dataset.yaml          # Config file
├── train/
│   ├── healthy/          # ~3,040 ảnh
│   │   ├── h0001.png
│   │   └── ...
│   ├── sick_but_no_tb/   # ~2,880 ảnh
│   │   └── ...
│   ├── active_tb/        # ~739 ảnh
│   │   └── ...
│   └── latent_tb/        # ~512 ảnh (sau augment)
│       └── ...
└── val/
    ├── healthy/          # ~760 ảnh
    ├── sick_but_no_tb/   # ~720 ảnh
    ├── active_tb/        # ~185 ảnh
    └── latent_tb/        # ~128 ảnh
```

### 3.4 File Config (dataset.yaml)

```yaml
names:
- healthy
- sick_but_no_tb
- active_tb
- latent_tb
nc: 4
path: D:\Workspace\ai4life\datasets\dataset_cls
train: train
val: val
```

### 3.5 Chia Train/Val

- **Phương pháp**: Stratified Split (giữ tỷ lệ lớp)
- **Tỷ lệ**: 80% Train / 20% Val
- **Random seed**: 42 (reproducibility)

```python
train_df, val_df = train_test_split(
    self.df, 
    test_size=0.2,
    stratify=self.df['class_name'],
    random_state=42
)
```

---

## 4. Stage 2: Detection Dataset

### 4.1 Mục Tiêu

Tạo dataset cho **YOLOv8** với nhiệm vụ phát hiện và định vị tổn thương TB:

```
┌─────────────────────────────────────────────────────────────┐
│                  2 LỚP PHÁT HIỆN                           │
├─────────────────┬──────────────────────────────────────────┤
│ Class 0         │ active_tb  - Tổn thương lao hoạt động    │
│ Class 1         │ latent_tb  - Tổn thương lao tiềm ẩn      │
└─────────────────┴──────────────────────────────────────────┘
```

### 4.2 Hai Loại Samples

#### 4.2.1 Positive Samples (Có Object)

| Thuộc tính | Mô tả |
|------------|-------|
| **Nguồn** | Ảnh `active_tb` và `latent_tb` |
| **Điều kiện** | Có bounding box trong CSV |
| **Label file** | Chứa tọa độ bbox định dạng YOLO |

#### 4.2.2 Background Samples (Không Có Object)

| Thuộc tính | Mô tả |
|------------|-------|
| **Nguồn** | Ảnh `healthy` và `sick_but_no_tb` |
| **Số lượng** | 500 ảnh/lớp (mặc định) |
| **Label file** | File `.txt` **RỖNG** |

> ⚠️ **TẠI SAO CẦN BACKGROUND IMAGES?**
> 
> - YOLO cần học phân biệt "có object" vs "không có object"
> - Nếu chỉ train với ảnh có TB → **False Positive cao!**
> - Background images dạy model: "Ảnh này KHÔNG có tổn thương TB"
> - File `.txt` rỗng = không có object nào trong ảnh này

### 4.3 Chuyển Đổi Bounding Box

#### Format Gốc (CSV):
```python
bbox = {'xmin': 180, 'ymin': 120, 'width': 150, 'height': 200}
```

#### Format YOLO:
```
<class_id> <x_center> <y_center> <width> <height>
```

Trong đó tất cả giá trị được **chuẩn hóa [0, 1]** theo kích thước ảnh.

#### Công Thức Chuyển Đổi:

```python
# Giả sử ảnh 512x512
img_width = 512
img_height = 512

# Tính tọa độ trung tâm (normalized)
x_center = (xmin + width/2) / img_width
y_center = (ymin + height/2) / img_height

# Tính kích thước (normalized)
width_norm = width / img_width
height_norm = height / img_height

# Clamp giá trị trong [0, 1]
x_center = max(0, min(1, x_center))
y_center = max(0, min(1, y_center))
width_norm = max(0, min(1, width_norm))
height_norm = max(0, min(1, height_norm))
```

### 4.4 Cấu Trúc Output

```
dataset_det/
├── dataset.yaml          # Config file
├── images/
│   ├── train/
│   │   ├── tb_0001.png   # Positive: active_tb
│   │   ├── ltb_0001.png  # Positive: latent_tb
│   │   ├── h0001.png     # Background: healthy
│   │   └── s0001.png     # Background: sick_but_no_tb
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── tb_0001.txt   # "0 0.5 0.4 0.3 0.4"
    │   ├── ltb_0001.txt  # "1 0.6 0.5 0.2 0.3"
    │   ├── h0001.txt     # (file rỗng - không có object)
    │   └── s0001.txt     # (file rỗng - không có object)
    └── val/
        └── ...
```

### 4.5 File Config (dataset.yaml)

```yaml
names:
- active_tb
- latent_tb
nc: 2
path: D:\Workspace\ai4life\datasets\dataset_det
train: images/train
val: images/val
```

### 4.6 Thống Kê Detection Dataset

```
📊 THỐNG KÊ DETECTION DATASET:
┌─────────────────────────────────────────────────────────────┐
│ TRAIN:                                                      │
│   active_tb (class 0):     739 ảnh                         │
│   latent_tb (class 1):     512 ảnh (sau augment)           │
│   background (no object):  800 ảnh                         │
│   ─────────────────────────────────                        │
│   Tổng:                    2,051 ảnh                       │
├─────────────────────────────────────────────────────────────┤
│ VAL:                                                        │
│   active_tb (class 0):     185 ảnh                         │
│   latent_tb (class 1):     128 ảnh                         │
│   background (no object):  200 ảnh                         │
│   ─────────────────────────────────                        │
│   Tổng:                    513 ảnh                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Kỹ Thuật Augmentation

### 5.1 Vấn Đề: Mất Cân Bằng Dữ Liệu

```
latent_tb: 239 ảnh  ←── Chỉ 2.7% dataset!
active_tb: 924 ảnh
Tỷ lệ mất cân bằng: ~3.9:1
```

### 5.2 Giải Pháp: Offline Augmentation

Tăng số lượng ảnh `latent_tb` **TRƯỚC KHI** chia train/val:

```python
self.offline_augmentation('latent_tb', target_count=800)
# 239 → 640+ ảnh
```

### 5.3 Các Kỹ Thuật Augmentation

Các kỹ thuật được thiết kế **phù hợp với ảnh X-quang y tế**:

| Kỹ thuật | Mô tả | Tham số |
|----------|-------|---------|
| **Rotation** | Xoay ảnh nhẹ | ±15° |
| **Brightness** | Thay đổi độ sáng | 0.7 - 1.3 |
| **Horizontal Flip** | Lật ngang ảnh | - |
| **Contrast** | Thay đổi độ tương phản | 0.8 - 1.2 |
| **Combined** | Kết hợp nhiều kỹ thuật | - |

### 5.4 Chi Tiết Implementation

```python
def _apply_augmentation(self, img: np.ndarray) -> tuple:
    aug_type = random.choice(['rotate', 'brightness', 'flip', 'contrast', 'combined'])
    
    if aug_type == 'rotate':
        # Xoay ±15 độ - mô phỏng góc chụp khác nhau
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
    elif aug_type == 'brightness':
        # Thay đổi độ sáng - mô phỏng điều kiện chụp khác nhau
        factor = random.uniform(0.7, 1.3)
        img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        
    elif aug_type == 'flip':
        # Lật ngang - vẫn giữ đặc điểm y học
        img = cv2.flip(img, 1)
        
    elif aug_type == 'contrast':
        # Thay đổi contrast
        factor = random.uniform(0.8, 1.2)
        mean = np.mean(img)
        img = cv2.convertScaleAbs(img, alpha=factor, beta=(1 - factor) * mean)
        
    elif aug_type == 'combined':
        # Kết hợp: flip (50%) + brightness + rotation nhẹ
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        factor = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        angle = random.uniform(-10, 10)
        # ... rotation code
    
    return img, aug_type
```

### 5.5 Xử Lý Bounding Box Khi Flip

Khi lật ngang ảnh, bounding box cũng cần được cập nhật:

```python
def _flip_bbox(self, bbox_str: str, img_width: int) -> str:
    bbox = ast.literal_eval(bbox_str)
    # Tính lại xmin sau khi flip
    new_xmin = img_width - bbox['xmin'] - bbox['width']
    bbox['xmin'] = new_xmin
    return str(bbox)
```

---

## 6. Cấu Trúc Thư Mục Output

```
datasets/
├── augmented_temp/           # Ảnh augmented tạm thời
│   └── latent_tb/
│       ├── ltb_001_aug_0_rotate.png
│       ├── ltb_001_aug_1_flip.png
│       └── ...
│
├── dataset_cls/              # CLASSIFICATION DATASET
│   ├── dataset.yaml
│   ├── train.cache           # Cache để train nhanh hơn
│   ├── val.cache
│   ├── train/
│   │   ├── healthy/
│   │   ├── sick_but_no_tb/
│   │   ├── active_tb/
│   │   └── latent_tb/
│   └── val/
│       ├── healthy/
│       ├── sick_but_no_tb/
│       ├── active_tb/
│       └── latent_tb/
│
└── dataset_det/              # DETECTION DATASET
    ├── dataset.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

---

## 7. Hướng Dẫn Sử Dụng

### 7.1 Command Line

```bash
# Chạy với tham số mặc định
python prepare_data.py \
    --csv tbx11k-simplified/data.csv \
    --images tbx11k-simplified/images \
    --output datasets

# Tùy chỉnh tham số
python prepare_data.py \
    --csv tbx11k-simplified/data.csv \
    --images tbx11k-simplified/images \
    --output datasets \
    --val-ratio 0.2 \
    --latent-target 800 \
    --bg-samples 500

# Không thực hiện augmentation
python prepare_data.py \
    --csv tbx11k-simplified/data.csv \
    --images tbx11k-simplified/images \
    --output datasets \
    --no-augment
```

### 7.2 Python Script

```python
from prepare_data import DataPreparer

# Khởi tạo
preparer = DataPreparer(
    csv_path='tbx11k-simplified/data.csv',
    images_dir='tbx11k-simplified/images',
    output_dir='datasets',
    val_ratio=0.2
)

# Chạy pipeline
preparer.run(
    augment_latent_tb=True,
    latent_target=800,
    bg_samples_per_class=500
)
```

### 7.3 Tham Số Có Thể Tùy Chỉnh

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--csv` | (bắt buộc) | Đường dẫn file CSV metadata |
| `--images` | (bắt buộc) | Thư mục chứa ảnh gốc |
| `--output` | `./datasets` | Thư mục output |
| `--val-ratio` | `0.2` | Tỷ lệ validation (20%) |
| `--no-augment` | `False` | Bỏ qua offline augmentation |
| `--latent-target` | `800` | Số ảnh latent_tb mục tiêu |
| `--bg-samples` | `500` | Số ảnh background mỗi lớp |

---

## 📈 Tóm Tắt

### So Sánh 2 Dataset

| Aspect | Classification | Detection |
|--------|---------------|-----------|
| **Mục đích** | Phân loại 4 loại X-quang | Định vị tổn thương TB |
| **Số lớp** | 4 | 2 |
| **Format** | Folder-based | YOLO txt format |
| **Tổng ảnh** | ~9,000+ | ~2,500+ |
| **Augmentation** | ✅ latent_tb | ✅ latent_tb |
| **Background** | Không cần | ✅ 1,000 ảnh |
| **Bbox** | Không cần | ✅ Chuẩn hóa |

### Quy Trình Xử Lý

1. **Load** → Đọc CSV và phân tích
2. **Augment** → Tăng cường latent_tb (239 → 640+)
3. **Split** → Chia 80/20 stratified
4. **Classification** → Copy ảnh vào folder theo class
5. **Detection** → Copy ảnh + tạo label YOLO format
6. **Config** → Tạo dataset.yaml

---

*Tài liệu được tạo: 01/12/2025*  
*Dự án: AI4Life - Hệ thống phát hiện Lao phổi*
