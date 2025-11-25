# Giải thích chi tiết: Quy trình xử lý dữ liệu trong Project TB Detection

## 📋 Tổng quan

Project này xử lý dữ liệu từ format CSV metadata (chứa thông tin về ảnh X-quang và bounding boxes) sang format YOLO (format chuẩn để training YOLO model). Quy trình bao gồm: đọc CSV, phân tích dataset, tìm và copy ảnh, chuyển đổi bounding boxes, và tạo cấu trúc thư mục YOLO.

---

## 🔄 Luồng xử lý dữ liệu tổng thể

```
INPUT: CSV file + Thư mục ảnh
    ↓
1. Đọc CSV metadata (parse_csv)
    ↓
2. Phân tích thống kê dataset (analyze_dataset)
    ↓
3. Tạo cấu trúc thư mục YOLO (create_directories)
    ↓
4. Chuyển đổi dữ liệu (convert_to_yolo_format)
    ├── Tìm và copy ảnh
    ├── Parse bounding boxes
    ├── Chuyển đổi sang YOLO format (normalized)
    └── Tạo file labels (.txt)
    ↓
5. Tạo file config YAML (create_yaml_config)
    ↓
OUTPUT: Dataset YOLO format + dataset.yaml
```

---

## 📊 Format dữ liệu đầu vào (CSV)

### Cấu trúc file CSV

File `data.csv` có các columns sau:

| Column | Mô tả | Ví dụ |
|--------|-------|-------|
| `fname` | Tên file ảnh | `h0001.png` |
| `image_height` | Chiều cao ảnh (pixels) | `512` |
| `image_width` | Chiều rộng ảnh (pixels) | `512` |
| `source` | Phân chia dataset | `train` hoặc `val` |
| `bbox` | Bounding box | `{'xmin': 100, 'ymin': 150, 'width': 200, 'height': 180}` hoặc `'none'` |
| `target` | Nhãn chính | `no_tb` hoặc `tb` |
| `tb_type` | Loại TB (tùy chọn) | `none` hoặc loại TB cụ thể |
| `image_type` | Loại ảnh | `healthy`, `sick_but_no_tb`, hoặc `tb` |

### Ví dụ dòng trong CSV:

```csv
fname,image_height,image_width,source,bbox,target,tb_type,image_type
h0001.png,512,512,train,none,no_tb,none,healthy
tb001.png,512,512,val,"{'xmin': 100, 'ymin': 150, 'width': 200, 'height': 180}",tb,active,tb
```

---

## 🔍 Chi tiết từng bước xử lý

### Bước 1: Khởi tạo TBDataPreprocessor

```python
preprocessor = TBDataPreprocessor(csv_path, images_dir, output_dir)
```

**Tham số:**
- `csv_path`: Đường dẫn đến file CSV metadata
- `images_dir`: Thư mục chứa các file ảnh gốc
- `output_dir`: Thư mục output để lưu dataset YOLO format

**Class mapping được định nghĩa:**
```python
self.class_map = {
    'healthy': 0,           # Class 0: Khỏe mạnh
    'sick_but_no_tb': 1,   # Class 1: Bệnh nhưng không phải lao
    'tb': 2                 # Class 2: Bệnh lao phổi
}
```

---

### Bước 2: Đọc CSV (parse_csv)

```python
def parse_csv(self):
    self.df = pd.read_csv(self.csv_path)
    # In thống kê cơ bản
    print(f"✅ Đã load {len(self.df)} samples")
    print(f"📊 Columns: {list(self.df.columns)}")
    print(f"📊 Phân bố target: {self.df['target'].value_counts().to_dict()}")
    print(f"📊 Phân bố image_type: {self.df['image_type'].value_counts().to_dict()}")
    print(f"📊 Phân bố source: {self.df['source'].value_counts().to_dict()}")
```

**Chức năng:**
- Đọc file CSV vào DataFrame pandas
- Hiển thị thống kê cơ bản:
  - Tổng số samples
  - Danh sách columns
  - Phân bố theo target (no_tb/tb)
  - Phân bố theo image_type (healthy/sick_but_no_tb/tb)
  - Phân bố theo source (train/val)

**Output:** DataFrame `self.df` chứa toàn bộ dữ liệu từ CSV

---

### Bước 3: Phân tích dataset (analyze_dataset)

```python
def analyze_dataset(self):
    # 1. Phân bố theo source (train/val)
    print(self.df['source'].value_counts())
    
    # 2. Phân bố theo target
    print(self.df['target'].value_counts())
    
    # 3. Phân bố theo image_type
    print(self.df['image_type'].value_counts())
    
    # 4. TB type distribution (nếu có)
    if 'tb_type' in self.df.columns:
        print(self.df[self.df['target'] == 'tb']['tb_type'].value_counts())
    
    # 5. Số lượng có bbox
    has_bbox = self.df['bbox'].apply(lambda x: x != 'none' and pd.notna(x)).sum()
    print(f"Số ảnh có bounding box: {has_bbox}/{len(self.df)}")
```

**Chức năng:**
- Phân tích thống kê chi tiết về dataset
- Giúp hiểu phân bố dữ liệu trước khi training
- Phát hiện các vấn đề như mất cân bằng dữ liệu

**Output:** In ra console các thống kê

---

### Bước 4: Tạo cấu trúc thư mục (create_directories)

```python
def create_directories(self):
    splits = ['train', 'val']
    
    for split in splits:
        os.makedirs(f'{self.output_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{self.output_dir}/labels/{split}', exist_ok=True)
```

**Chức năng:**
- Tạo cấu trúc thư mục theo format YOLO:
  ```
  output_dir/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
  ```

**Output:** Cấu trúc thư mục YOLO đã được tạo

---

### Bước 5: Chuyển đổi sang YOLO format (convert_to_yolo_format)

Đây là bước quan trọng nhất, xử lý từng dòng trong CSV:

#### 5.1. Tìm đường dẫn ảnh (find_image_path)

```python
def find_image_path(self, filename):
    possible_paths = [
        os.path.join(self.images_dir, filename),
        os.path.join(self.images_dir, 'train', filename),
        os.path.join(self.images_dir, 'val', filename),
        os.path.join(self.images_dir, 'test', filename),
        os.path.join(os.path.dirname(self.images_dir), 'test', filename),
        filename,  # Đường dẫn trực tiếp
    ]
    
    # Thử từng đường dẫn
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Tìm kiếm đệ quy nếu không tìm thấy
    for root, dirs, files in os.walk(self.images_dir):
        if filename in files:
            return os.path.join(root, filename)
    
    return None
```

**Chức năng:**
- Tìm file ảnh trong nhiều vị trí có thể
- Tìm kiếm đệ quy nếu không tìm thấy ở các vị trí thông thường
- Trả về `None` nếu không tìm thấy

**Lý do:** Dataset có thể có cấu trúc thư mục khác nhau, cần linh hoạt tìm kiếm

---

#### 5.2. Parse bounding box (safe_parse_bbox)

```python
def safe_parse_bbox(self, bbox_str):
    try:
        # Kiểm tra nếu không có bbox
        if pd.isna(bbox_str) or bbox_str == 'none' or bbox_str == '':
            return None
        
        # Nếu là string, parse như dictionary
        if isinstance(bbox_str, str):
            bbox_dict = ast.literal_eval(bbox_str)
            return bbox_dict
        else:
            return bbox_str
    except Exception as e:
        print(f"⚠️ Lỗi parse bbox: {bbox_str} - {e}")
        return None
```

**Chức năng:**
- Parse string bbox thành dictionary an toàn
- Xử lý các trường hợp: `'none'`, `NaN`, string rỗng
- Sử dụng `ast.literal_eval()` để parse string thành dict an toàn

**Input:** String như `"{'xmin': 100, 'ymin': 150, 'width': 200, 'height': 180}"`

**Output:** Dictionary `{'xmin': 100, 'ymin': 150, 'width': 200, 'height': 180}` hoặc `None`

---

#### 5.3. Chuyển đổi bounding box sang YOLO format

**Format gốc (Absolute coordinates):**
```
xmin, ymin, width, height
Ví dụ: xmin=100, ymin=150, width=200, height=180
```

**Format YOLO (Normalized center coordinates):**
```
class_id x_center y_center width height
Tất cả giá trị đều normalized (0-1)
```

**Công thức chuyển đổi:**

```python
# Lấy kích thước ảnh
img_width = float(row['image_width'])   # Ví dụ: 512
img_height = float(row['image_height'])  # Ví dụ: 512

# Lấy tọa độ bbox gốc
xmin = float(bbox_data['xmin'])         # Ví dụ: 100
ymin = float(bbox_data['ymin'])         # Ví dụ: 150
bbox_width = float(bbox_data['width'])  # Ví dụ: 200
bbox_height = float(bbox_data['height']) # Ví dụ: 180

# Tính tâm bbox
x_center_abs = xmin + bbox_width/2      # 100 + 200/2 = 200
y_center_abs = ymin + bbox_height/2     # 150 + 180/2 = 240

# Normalize về [0, 1]
x_center = x_center_abs / img_width     # 200 / 512 = 0.390625
y_center = y_center_abs / img_height    # 240 / 512 = 0.46875
width = bbox_width / img_width          # 200 / 512 = 0.390625
height = bbox_height / img_height       # 180 / 512 = 0.3515625

# Đảm bảo giá trị trong [0, 1]
x_center = max(0, min(1, x_center))
y_center = max(0, min(1, y_center))
width = max(0, min(1, width))
height = max(0, min(1, height))
```

**Ví dụ cụ thể:**

```
Input (Absolute):
  Ảnh: 512x512
  Bbox: xmin=100, ymin=150, width=200, height=180

Tính toán:
  x_center = (100 + 200/2) / 512 = 200/512 = 0.390625
  y_center = (150 + 180/2) / 512 = 240/512 = 0.46875
  width = 200 / 512 = 0.390625
  height = 180 / 512 = 0.3515625

Output (YOLO format):
  2 0.390625 0.46875 0.390625 0.3515625
  (class_id=2 là 'tb')
```

---

#### 5.4. Xác định class_id

```python
# Map class dựa trên target và image_type
if row['target'] == 'tb':
    class_id = 2  # tb
elif row['image_type'] == 'healthy':
    class_id = 0  # healthy
else:
    class_id = 1  # sick_but_no_tb
```

**Logic:**
- Nếu `target == 'tb'` → Class 2 (tb)
- Nếu `image_type == 'healthy'` → Class 0 (healthy)
- Còn lại → Class 1 (sick_but_no_tb)

---

#### 5.5. Ghi file label

```python
# Tạo tên file label
label_filename = filename.replace('.png', '.txt')
label_path = os.path.join(self.output_dir, 'labels', split, label_filename)

# Ghi vào file
with open(label_path, 'a') as f:
    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
```

**Chức năng:**
- Tạo file `.txt` tương ứng với mỗi ảnh
- Format: `class_id x_center y_center width height`
- Sử dụng `'a'` (append) để hỗ trợ nhiều bbox trong 1 ảnh

**Ví dụ file label:**
```
2 0.390625 0.46875 0.390625 0.3515625
2 0.650000 0.300000 0.200000 0.250000
```
(Ảnh này có 2 vùng bệnh lao)

**Trường hợp không có bbox:**
```python
else:
    # Tạo file label rỗng cho ảnh healthy/sick_but_no_tb
    with open(label_path, 'w') as f:
        pass
```
Tạo file `.txt` rỗng cho ảnh không có bbox (healthy hoặc sick_but_no_tb)

---

#### 5.6. Copy ảnh

```python
src_image = self.find_image_path(filename)
dst_image = os.path.join(self.output_dir, 'images', split, filename)

if src_image is not None and os.path.exists(src_image):
    shutil.copy2(src_image, dst_image)
else:
    missing_images.append(filename)
    skipped_count += 1
    continue
```

**Chức năng:**
- Copy ảnh từ thư mục gốc sang thư mục YOLO format
- Sử dụng `shutil.copy2()` để giữ nguyên metadata
- Lưu danh sách ảnh không tìm thấy

---

### Bước 6: Tạo file config YAML (create_yaml_config)

```python
def create_yaml_config(self):
    yaml_content = f"""# TBX11K Dataset Configuration
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val

# Classes
nc: 3
names: ['healthy', 'sick_but_no_tb', 'tb']

# Dataset info
roboflow:
  workspace: tbx11k
  project: tb-detection
  version: 1
"""
    
    yaml_path = os.path.join(self.output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path
```

**Chức năng:**
- Tạo file `dataset.yaml` theo format YOLO
- File này chứa:
  - Đường dẫn dataset
  - Đường dẫn train/val
  - Số lượng classes (nc: 3)
  - Tên các classes

**Output:** File `dataset.yaml` trong thư mục output

**Ví dụ nội dung:**
```yaml
# TBX11K Dataset Configuration
path: D:\Workspace\ai4life\tbx11k-simplified
train: images/train
val: images/val

# Classes
nc: 3
names: ['healthy', 'sick_but_no_tb', 'tb']

# Dataset info
roboflow:
  workspace: tbx11k
  project: tb-detection
  version: 1
```

---

## 📁 Cấu trúc dữ liệu đầu ra

Sau khi xử lý, cấu trúc thư mục sẽ như sau:

```
output_dir/
├── dataset.yaml                    # File config YOLO
├── images/
│   ├── train/
│   │   ├── h0001.png
│   │   ├── h0002.png
│   │   └── ...
│   └── val/
│       ├── h1001.png
│       └── ...
├── labels/
│   ├── train/
│   │   ├── h0001.txt              # Label tương ứng (có thể rỗng)
│   │   ├── h0002.txt
│   │   └── ...
│   └── val/
│       ├── h1001.txt
│       └── ...
└── missing_images.txt             # Danh sách ảnh không tìm thấy (nếu có)
```

---

## 📝 Format file label YOLO

### File label có bbox:
```
2 0.390625 0.46875 0.390625 0.3515625
2 0.650000 0.300000 0.200000 0.250000
```

**Giải thích:**
- Dòng 1: Class 2 (tb), tâm tại (0.390625, 0.46875), kích thước 0.390625 x 0.3515625
- Dòng 2: Class 2 (tb), tâm tại (0.65, 0.3), kích thước 0.2 x 0.25

### File label không có bbox (healthy/sick_but_no_tb):
```
(rỗng - file tồn tại nhưng không có nội dung)
```

---

## 🔄 Quy trình chạy (run)

```python
def run(self):
    self.parse_csv()                    # Bước 1: Đọc CSV
    self.analyze_dataset()              # Bước 2: Phân tích
    self.create_directories()           # Bước 3: Tạo thư mục
    self.convert_to_yolo_format()       # Bước 4: Chuyển đổi
    yaml_path = self.create_yaml_config() # Bước 5: Tạo YAML
    return yaml_path
```

**Chức năng:**
- Chạy toàn bộ pipeline preprocessing
- Trả về đường dẫn file `dataset.yaml`

---

## ⚠️ Xử lý lỗi và edge cases

### 1. Ảnh không tìm thấy
- Lưu vào danh sách `missing_images`
- Tạo file `missing_images.txt` để kiểm tra sau
- Bỏ qua và tiếp tục xử lý ảnh khác

### 2. Bbox không parse được
- In cảnh báo và bỏ qua
- Tạo file label rỗng nếu không có bbox

### 3. Giá trị bbox ngoài phạm vi
- Sử dụng `max(0, min(1, value))` để đảm bảo trong [0, 1]

### 4. Nhiều bbox trong 1 ảnh
- Sử dụng mode `'a'` (append) khi ghi file
- Mỗi bbox là 1 dòng trong file label

---

## 📊 Thống kê sau khi xử lý

Sau khi chạy, sẽ hiển thị:
```
✅ Hoàn tất!
   - Converted: 8500        # Số ảnh đã chuyển đổi thành công
   - Skipped: 303          # Số ảnh bị bỏ qua (lỗi hoặc thiếu)
   - Missing images saved to: output_dir/missing_images.txt
```

---

## 🎯 Tóm tắt

1. **Input**: CSV file + Thư mục ảnh
2. **Xử lý**:
   - Đọc và phân tích CSV
   - Tìm và copy ảnh
   - Parse và chuyển đổi bounding boxes
   - Tạo file labels
3. **Output**: Dataset YOLO format + dataset.yaml

**Điểm quan trọng:**
- Bounding boxes được normalize về [0, 1]
- Format YOLO: `class_id x_center y_center width height`
- Ảnh không có bbox vẫn có file label (rỗng)
- Hỗ trợ nhiều bbox trong 1 ảnh

---

## 💡 Lưu ý

1. **Normalization**: Tất cả tọa độ đều được normalize để không phụ thuộc vào kích thước ảnh
2. **Class mapping**: Dựa trên cả `target` và `image_type` để xác định class
3. **Flexible path finding**: Tìm ảnh ở nhiều vị trí có thể để xử lý các cấu trúc dataset khác nhau
4. **Error handling**: Xử lý lỗi một cách graceful, không dừng toàn bộ pipeline

