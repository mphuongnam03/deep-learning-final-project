# Tại sao cần chuyển đổi dataset sang YOLO format?

## 📋 Tổng quan

YOLO (You Only Look Once) là một framework object detection yêu cầu dữ liệu ở format đặc biệt. Format này khác hoàn toàn với format dataset thông thường (CSV, JSON, XML). Việc chuyển đổi là bắt buộc để YOLO có thể đọc và sử dụng dữ liệu để training.

---

## ❓ Tại sao YOLO cần format riêng?

### 1. **YOLO được thiết kế với format cụ thể**

YOLO framework (Ultralytics) được xây dựng để đọc dữ liệu theo một cấu trúc nhất định:

- Ảnh và labels phải được tổ chức theo cấu trúc thư mục cụ thể
- Labels phải ở dạng text file với format chuẩn
- Tọa độ phải được normalize (0-1)

**Nếu không chuyển đổi:** YOLO sẽ không thể đọc và sử dụng dataset của bạn.

### 2. **YOLO sử dụng normalized coordinates**

YOLO yêu cầu tọa độ bounding box phải được normalize về khoảng [0, 1] thay vì sử dụng pixel coordinates tuyệt đối. Điều này giúp:

- Model không phụ thuộc vào kích thước ảnh
- Dễ dàng resize ảnh trong quá trình training
- Tăng tốc độ tính toán

### 3. **Cấu trúc thư mục chuẩn**

YOLO yêu cầu cấu trúc thư mục cụ thể:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## 🔄 So sánh Format gốc vs YOLO Format

### Format gốc (Dataset thông thường)

#### 1. **Cấu trúc dữ liệu:**

```
Dataset gốc:
├── data.csv (chứa tất cả metadata)
├── images/ (ảnh có thể ở bất kỳ đâu)
└── (không có labels riêng)
```

#### 2. **Format bounding box trong CSV:**

```csv
fname,image_height,image_width,source,bbox,target,image_type
tb001.png,512,512,train,"{'xmin': 100, 'ymin': 150, 'width': 200, 'height': 180}",tb,tb
```

**Đặc điểm:**

- ✅ Tất cả thông tin trong 1 file CSV
- ✅ Bounding box: **Absolute coordinates** (pixel)
- ✅ Format: `xmin, ymin, width, height`
- ✅ Dễ đọc và chỉnh sửa bằng Excel/CSV editor
- ❌ YOLO không thể đọc trực tiếp

#### 3. **Ví dụ cụ thể:**

**Input (Format gốc):**

```
Ảnh: tb001.png (512x512 pixels)
Bbox trong CSV: {'xmin': 100, 'ymin': 150, 'width': 200, 'height': 180}
```

**Vấn đề:**

- Tọa độ là **absolute** (phụ thuộc kích thước ảnh)
- Nếu resize ảnh → phải tính lại tọa độ
- Không có file label riêng cho từng ảnh

---

### YOLO Format

#### 1. **Cấu trúc dữ liệu:**

```
Dataset YOLO:
├── dataset.yaml (config file)
├── images/
│   ├── train/
│   │   ├── tb001.png
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── tb001.txt  ← File label riêng
    │   └── ...
    └── val/
        └── ...
```

#### 2. **Format bounding box trong file .txt:**

```
2 0.390625 0.46875 0.390625 0.3515625
```

**Đặc điểm:**

- ✅ Mỗi ảnh có 1 file label riêng (.txt)
- ✅ Bounding box: **Normalized coordinates** (0-1)
- ✅ Format: `class_id x_center y_center width height`
- ✅ Không phụ thuộc kích thước ảnh
- ✅ YOLO có thể đọc và sử dụng trực tiếp

#### 3. **Ví dụ cụ thể:**

**Output (YOLO Format):**

```
Ảnh: images/train/tb001.png (512x512 pixels)
Label: labels/train/tb001.txt
Nội dung: 2 0.390625 0.46875 0.390625 0.3515625
```

**Giải thích:**

- `2`: Class ID (tb = 2)
- `0.390625`: x_center normalized (200/512)
- `0.46875`: y_center normalized (240/512)
- `0.390625`: width normalized (200/512)
- `0.3515625`: height normalized (180/512)

---

## 📊 Bảng so sánh chi tiết

| Đặc điểm | Format gốc (CSV) | YOLO Format |
|----------|------------------|------------|
| **Cấu trúc** | 1 file CSV chứa tất cả | Mỗi ảnh có 1 file label |
| **Tọa độ** | Absolute (pixels) | Normalized (0-1) |
| **Format bbox** | `xmin, ymin, width, height` | `x_center, y_center, width, height` |
| **Phụ thuộc kích thước** | ✅ Có | ❌ Không |
| **Resize ảnh** | Phải tính lại tọa độ | Không cần |
| **YOLO đọc được** | ❌ Không | ✅ Có |
| **Dễ chỉnh sửa** | ✅ Dễ (Excel) | ⚠️ Khó hơn (text editor) |
| **Tốc độ đọc** | Chậm (phải parse CSV) | Nhanh (đọc trực tiếp) |

---

## 🔍 Ví dụ chuyển đổi cụ thể

### Input (Format gốc)

**File CSV:**

```csv
fname,image_height,image_width,source,bbox,target,image_type
tb001.png,512,512,train,"{'xmin': 100, 'ymin': 150, 'width': 200, 'height': 180}",tb,tb
```

**Giải thích:**

- Ảnh: 512x512 pixels
- Bbox: xmin=100, ymin=150, width=200, height=180
- Vị trí: Góc trên trái tại (100, 150), kích thước 200x180

### Quá trình chuyển đổi

```python
# 1. Tính tâm bbox
x_center_abs = 100 + 200/2 = 200
y_center_abs = 150 + 180/2 = 240

# 2. Normalize về [0, 1]
x_center = 200 / 512 = 0.390625
y_center = 240 / 512 = 0.46875
width = 200 / 512 = 0.390625
height = 180 / 512 = 0.3515625

# 3. Xác định class_id
class_id = 2  # (tb)
```

### Output (YOLO Format)

**File: `labels/train/tb001.txt`**

```
2 0.390625 0.46875 0.390625 0.3515625
```

**Giải thích:**

- `2`: Class ID (tb)
- `0.390625`: Tâm X normalized
- `0.46875`: Tâm Y normalized
- `0.390625`: Chiều rộng normalized
- `0.3515625`: Chiều cao normalized

---

## 💡 Lợi ích của YOLO Format

### 1. **Không phụ thuộc kích thước ảnh**

**Format gốc:**

```
Ảnh 512x512: bbox = (100, 150, 200, 180)
Ảnh 1024x1024: bbox = (200, 300, 400, 360) ← Phải tính lại!
```

**YOLO Format:**

```
Ảnh 512x512: bbox = (0.390625, 0.46875, 0.390625, 0.3515625)
Ảnh 1024x1024: bbox = (0.390625, 0.46875, 0.390625, 0.3515625) ← Giống nhau!
```

**Lợi ích:** Có thể resize ảnh mà không cần chỉnh sửa labels.

### 2. **Tốc độ đọc nhanh hơn**

**Format gốc:**

- Phải đọc toàn bộ CSV
- Parse từng dòng để tìm label của ảnh
- Chậm với dataset lớn

**YOLO Format:**

- Đọc trực tiếp file label tương ứng
- Không cần parse CSV
- Nhanh hơn nhiều

### 3. **Tương thích với YOLO framework**

YOLO framework được thiết kế để:

- Đọc file `.txt` trong thư mục `labels/`
- Sử dụng normalized coordinates
- Tự động load ảnh từ thư mục `images/`

**Nếu không chuyển đổi:** YOLO sẽ không thể training!

### 4. **Hỗ trợ nhiều bbox trong 1 ảnh**

**Format gốc:**

```csv
fname,bbox
img1.png,"[{'xmin': 100, 'ymin': 150, 'width': 200, 'height': 180}, {'xmin': 300, 'ymin': 200, 'width': 150, 'height': 120}]"
```

→ Phức tạp khi parse

**YOLO Format:**

```
# labels/train/img1.txt
2 0.390625 0.46875 0.390625 0.3515625
2 0.5859375 0.390625 0.29296875 0.234375
```

→ Mỗi dòng là 1 bbox, dễ đọc

### 5. **Chuẩn hóa dữ liệu**

Tất cả dataset YOLO đều có cùng format:

- Cùng cấu trúc thư mục
- Cùng format label
- Dễ chia sẻ và sử dụng lại

---

## 🎯 Tại sao không dùng format gốc trực tiếp?

### Vấn đề 1: YOLO không đọc được CSV

YOLO framework chỉ đọc:

- File `.txt` trong thư mục `labels/`
- Format: `class_id x_center y_center width height`

**Không thể:**

- Đọc trực tiếp từ CSV
- Parse JSON/XML
- Đọc từ database

### Vấn đề 2: Absolute coordinates gây khó khăn

**Ví dụ:**

```
Ảnh gốc: 512x512
Bbox: (100, 150, 200, 180)

Khi resize về 256x256:
Bbox mới: (50, 75, 100, 90) ← Phải tính lại!
```

**Với normalized:**

```
Bbox: (0.390625, 0.46875, 0.390625, 0.3515625)

Khi resize về 256x256:
Bbox: (0.390625, 0.46875, 0.390625, 0.3515625) ← Không đổi!
```

### Vấn đề 3: Hiệu suất kém

- Đọc CSV mỗi lần cần label → Chậm
- Parse string bbox → Tốn thời gian
- Không tận dụng được caching

---

## 📝 Format chi tiết

### Format gốc (Absolute)

```
Bounding Box:
- xmin: Tọa độ X góc trên trái (pixels)
- ymin: Tọa độ Y góc trên trái (pixels)
- width: Chiều rộng (pixels)
- height: Chiều cao (pixels)

Ví dụ: xmin=100, ymin=150, width=200, height=180
```

### YOLO Format (Normalized)

```
Bounding Box:
- class_id: ID của class (0, 1, 2, ...)
- x_center: Tọa độ X của tâm (0-1)
- y_center: Tọa độ Y của tâm (0-1)
- width: Chiều rộng normalized (0-1)
- height: Chiều cao normalized (0-1)

Ví dụ: 2 0.390625 0.46875 0.390625 0.3515625
```

**Công thức chuyển đổi:**

```python
x_center = (xmin + width/2) / img_width
y_center = (ymin + height/2) / img_height
width_norm = width / img_width
height_norm = height / img_height
```

---

## 🔄 Quy trình chuyển đổi trong project

```
1. Đọc CSV
   ↓
2. Với mỗi dòng:
   ├── Tìm file ảnh
   ├── Copy ảnh vào images/train hoặc images/val
   ├── Parse bbox từ CSV
   ├── Chuyển đổi: absolute → normalized
   ├── Xác định class_id
   └── Ghi vào file .txt
   ↓
3. Tạo dataset.yaml
   ↓
4. Hoàn tất!
```

---

## ✅ Kết luận

### Tại sao cần chuyển đổi?

1. **YOLO yêu cầu format cụ thể** - Không thể đọc CSV trực tiếp
2. **Normalized coordinates** - Không phụ thuộc kích thước ảnh
3. **Cấu trúc thư mục chuẩn** - YOLO cần cấu trúc cụ thể
4. **Hiệu suất tốt hơn** - Đọc file riêng nhanh hơn parse CSV
5. **Tương thích framework** - YOLO được thiết kế cho format này

### Sự khác biệt chính

| Format gốc | YOLO Format |
|------------|------------|
| CSV metadata | File label riêng |
| Absolute coordinates | Normalized coordinates |
| `xmin, ymin, width, height` | `x_center, y_center, width, height` |
| Phụ thuộc kích thước | Không phụ thuộc |
| YOLO không đọc được | YOLO đọc được |

**→ Vì vậy, chuyển đổi là BẮT BUỘC để training YOLO model!**

---

## 💡 Lưu ý

1. **Normalization là quan trọng**: Giúp model học tốt hơn và không phụ thuộc kích thước ảnh
2. **Cấu trúc thư mục**: YOLO yêu cầu cấu trúc cụ thể, không thể tùy ý
3. **File label riêng**: Mỗi ảnh cần 1 file label tương ứng
4. **Format chuẩn**: Tất cả dataset YOLO đều dùng format này, dễ chia sẻ và tái sử dụng

