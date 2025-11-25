# Giải thích chi tiết các tham số training YOLO (train.py:34-87)

## 📋 Tổng quan

File `train.py` sử dụng hàm `model.train()` của Ultralytics YOLO với nhiều tham số để điều chỉnh quá trình huấn luyện. Tài liệu này giải thích chi tiết từng tham số và ảnh hưởng của chúng.

---

## 🔧 Các tham số cơ bản

### 1. `data=self.data_yaml`

```python
data=self.data_yaml
```

**Mô tả:** Đường dẫn đến file cấu hình dataset (dataset.yaml)

**Giá trị:** Đường dẫn file YAML (ví dụ: `'tbx11k-simplified/dataset.yaml'`)

**Chức năng:**

- YOLO đọc file này để biết:
  - Đường dẫn dataset
  - Thư mục train/val
  - Số lượng classes
  - Tên các classes

**Ví dụ nội dung dataset.yaml:**

```yaml
path: tbx11k-simplified
train: images/train
val: images/val
nc: 3
names: ['healthy', 'sick_but_no_tb', 'tb']
```

---

### 2. `epochs=epochs`

```python
epochs=epochs
```

**Mô tả:** Số lần model đi qua toàn bộ dataset

**Giá trị:** Số nguyên (mặc định: 100)

**Giải thích:**

- 1 epoch = 1 lần model học qua tất cả ảnh trong training set
- Epochs càng nhiều → Model học lâu hơn, có thể tốt hơn nhưng cũng có thể overfit

**Ví dụ:**

- `epochs=50`: Model học 50 lần qua dataset
- `epochs=100`: Model học 100 lần qua dataset

**Lưu ý:** Có thể dừng sớm nhờ `patience` (early stopping)

---

### 3. `imgsz=img_size`

```python
imgsz=img_size
```

**Mô tả:** Kích thước ảnh đầu vào (pixels)

**Giá trị:** Số nguyên (mặc định: 512)

**Chức năng:**

- Tất cả ảnh sẽ được resize về kích thước này trước khi đưa vào model
- Kích thước lớn hơn → Độ chính xác tốt hơn nhưng chậm hơn và tốn bộ nhớ hơn

**Ví dụ:**

- `imgsz=512`: Ảnh 512x512 pixels
- `imgsz=640`: Ảnh 640x640 pixels (YOLO mặc định)
- `imgsz=1024`: Ảnh 1024x1024 pixels (rất chậm, tốn bộ nhớ)

**Trade-off:**

- ✅ Kích thước lớn: Độ chính xác cao hơn
- ❌ Kích thước lớn: Chậm hơn, tốn bộ nhớ hơn

---

### 4. `batch=batch_size`

```python
batch=batch_size
```

**Mô tả:** Số lượng ảnh xử lý cùng lúc trong mỗi batch

**Giá trị:** Số nguyên (mặc định: 16)

**Chức năng:**

- Batch size = số ảnh được đưa vào model cùng lúc
- Batch lớn hơn → Training ổn định hơn, nhanh hơn nhưng tốn bộ nhớ hơn

**Ví dụ:**

- `batch=8`: Xử lý 8 ảnh/lần (tiết kiệm bộ nhớ)
- `batch=16`: Xử lý 16 ảnh/lần (cân bằng)
- `batch=32`: Xử lý 32 ảnh/lần (nhanh hơn nhưng tốn bộ nhớ)

**Lưu ý:**

- Batch size phụ thuộc vào VRAM của GPU
- Nếu hết bộ nhớ, giảm batch size hoặc `imgsz`

---

### 5. `device=device`

```python
device=device
```

**Mô tả:** Thiết bị để training (CPU hoặc GPU)

**Giá trị:** 

- `0`: GPU đầu tiên
- `1`: GPU thứ hai
- `'cpu'`: CPU (rất chậm)
- `[0, 1]`: Nhiều GPU

**Mặc định:** `0` (GPU đầu tiên)

**Lưu ý:** Training trên CPU rất chậm, nên dùng GPU nếu có

---

## 📁 Project Settings

### 6. `project=self.project`

```python
project=self.project
```

**Mô tả:** Tên thư mục project để lưu kết quả

**Giá trị:** String (mặc định: `'tb_detection'`)

**Chức năng:**

- Tất cả kết quả training sẽ được lưu trong thư mục này
- Kết quả bao gồm: weights, plots, metrics, logs

**Ví dụ:** Kết quả lưu trong `tb_detection/yolov8n_tb_20251011_190341/`

---

### 7. `name=f'yolov8{self.model_size}_tb_{datetime.now().strftime("%Y%m%d_%H%M%S")}'`

```python
name=f'yolov8{self.model_size}_tb_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
```

**Mô tả:** Tên thư mục cho lần training này

**Giá trị:** String với timestamp

**Ví dụ:** `'yolov8n_tb_20251011_190341'`

**Giải thích:**

- `yolov8n`: Model size (nano)
- `tb`: Tên project
- `20251011_190341`: Ngày giờ (YYYYMMDD_HHMMSS)

**Lợi ích:** Mỗi lần training có thư mục riêng, không bị ghi đè

---

### 8. `exist_ok=True`

```python
exist_ok=True
```

**Mô tả:** Cho phép ghi đè thư mục nếu đã tồn tại

**Giá trị:** Boolean (mặc định: `True`)

**Chức năng:**

- `True`: Nếu thư mục đã tồn tại, ghi đè
- `False`: Nếu thư mục đã tồn tại, báo lỗi

---

## ⚙️ Training Settings

### 9. `patience=20`

```python
patience=20  # Early stopping
```

**Mô tả:** Số epochs chờ đợi trước khi dừng sớm (Early Stopping)

**Giá trị:** Số nguyên (mặc định: 20)

**Chức năng:**

- Nếu mAP không cải thiện trong 20 epochs liên tiếp → Dừng training
- Tránh overfitting và tiết kiệm thời gian

**Ví dụ:**

- Epoch 50: mAP = 0.75
- Epoch 51-70: mAP không tăng → Dừng ở epoch 70
- Model tốt nhất được lưu ở epoch 50

**Lưu ý:** Nếu `patience=0`, không có early stopping

---

### 10. `save=True`

```python
save=True
```

**Mô tả:** Có lưu model weights không

**Giá trị:** Boolean (mặc định: `True`)

**Chức năng:**

- `True`: Lưu `best.pt` (model tốt nhất) và `last.pt` (model cuối cùng)
- `False`: Không lưu weights

---

### 11. `save_period=10`

```python
save_period=10  # Save mỗi 10 epochs
```

**Mô tả:** Lưu checkpoint mỗi N epochs

**Giá trị:** Số nguyên (mặc định: 10)

**Chức năng:**

- Mỗi 10 epochs, lưu thêm file `epochN.pt`
- Giúp có thể resume training từ checkpoint bất kỳ

**Ví dụ:**

- Epoch 10: Lưu `epoch10.pt`
- Epoch 20: Lưu `epoch20.pt`
- Epoch 30: Lưu `epoch30.pt`

---

## 🎯 Optimization Parameters

### 12. `optimizer='SGD'`

```python
optimizer='SGD'
```

**Mô tả:** Thuật toán tối ưu hóa

**Giá trị:** `'SGD'` hoặc `'Adam'` hoặc `'AdamW'`

**Giải thích:**

- **SGD (Stochastic Gradient Descent):** 
  - Mặc định cho YOLO
  - Ổn định, phù hợp với object detection
  - Cần điều chỉnh learning rate cẩn thận
  
- **Adam:**
  - Tự động điều chỉnh learning rate
  - Hội tụ nhanh hơn nhưng có thể không ổn định

**Khuyến nghị:** Dùng `SGD` cho YOLO (đã được tối ưu)

---

### 13. `lr0=0.01`

```python
lr0=0.01
```

**Mô tả:** Learning rate ban đầu (Initial Learning Rate)

**Giá trị:** Float (mặc định: 0.01)

**Giải thích:**

- Learning rate = Bước nhảy của model khi học
- `lr0` = Learning rate ở epoch đầu tiên
- Learning rate sẽ giảm dần theo thời gian (xem `lrf`)

**Ví dụ:**

- `lr0=0.01`: Bước nhảy ban đầu = 0.01
- `lr0=0.001`: Bước nhảy nhỏ hơn → Học chậm hơn, ổn định hơn
- `lr0=0.1`: Bước nhảy lớn hơn → Học nhanh hơn nhưng có thể không ổn định

**Lưu ý:**

- Learning rate quá lớn → Model không hội tụ
- Learning rate quá nhỏ → Học quá chậm

---

### 14. `lrf=0.01`

```python
lrf=0.01
```

**Mô tả:** Learning rate cuối cùng (Final Learning Rate Factor)

**Giá trị:** Float (mặc định: 0.01)

**Giải thích:**

- Learning rate sẽ giảm từ `lr0` xuống `lr0 * lrf`
- `lrf=0.01` → Learning rate cuối = `0.01 * 0.01 = 0.0001`

**Ví dụ:**

- `lr0=0.01`, `lrf=0.01` → LR cuối = 0.0001
- `lr0=0.01`, `lrf=0.1` → LR cuối = 0.001 (giảm ít hơn)

**Lợi ích:** Giảm learning rate giúp model fine-tune tốt hơn ở cuối training

---

### 15. `momentum=0.937`

```python
momentum=0.937
```

**Mô tả:** Momentum cho SGD optimizer

**Giá trị:** Float từ 0 đến 1 (mặc định: 0.937)

**Giải thích:**

- Momentum giúp SGD vượt qua các local minima
- Giá trị cao → Model "nhớ" hướng di chuyển trước đó
- Giá trị thấp → Model di chuyển chậm hơn, ổn định hơn

**Ví dụ:**

- `momentum=0.9`: Momentum cao, di chuyển nhanh
- `momentum=0.5`: Momentum thấp, di chuyển chậm

---

### 16. `weight_decay=0.0005`

```python
weight_decay=0.0005
```

**Mô tả:** L2 regularization (tránh overfitting)

**Giá trị:** Float (mặc định: 0.0005)

**Giải thích:**

- Weight decay = Penalty cho weights lớn
- Giúp model không học quá khớp với training data
- Giá trị cao → Model đơn giản hơn, ít overfit hơn

**Ví dụ:**

- `weight_decay=0.0005`: Regularization vừa phải
- `weight_decay=0.001`: Regularization mạnh hơn
- `weight_decay=0.0001`: Regularization yếu hơn

---

### 17. `warmup_epochs=3.0`

```python
warmup_epochs=3.0
```

**Mô tả:** Số epochs để "khởi động" learning rate

**Giá trị:** Float (mặc định: 3.0)

**Giải thích:**

- Trong 3 epochs đầu, learning rate tăng dần từ 0 lên `lr0`
- Giúp model ổn định ở đầu training
- Tránh gradient quá lớn làm model "nhảy" quá xa

**Ví dụ:**

- Epoch 1: LR = 0.003 (30% của lr0)
- Epoch 2: LR = 0.007 (70% của lr0)
- Epoch 3: LR = 0.01 (100% của lr0)
- Epoch 4+: LR giảm dần

---

### 18. `warmup_momentum=0.8`

```python
warmup_momentum=0.8
```

**Mô tả:** Momentum trong giai đoạn warmup

**Giá trị:** Float (mặc định: 0.8)

**Giải thích:**

- Momentum thấp hơn trong warmup để ổn định hơn
- Sau warmup, momentum tăng lên `momentum` (0.937)

---

### 19. `warmup_bias_lr=0.1`

```python
warmup_bias_lr=0.1
```

**Mô tả:** Learning rate cho bias trong warmup

**Giá trị:** Float (mặc định: 0.1)

**Giải thích:**

- Bias có learning rate riêng trong warmup
- Giúp bias học nhanh hơn ở đầu training

---

## 🎨 Data Augmentation Parameters

Data augmentation = Tạo thêm dữ liệu từ dữ liệu gốc bằng cách biến đổi ảnh

### 20. `hsv_h=0.015`

```python
hsv_h=0.015
```

**Mô tả:** Thay đổi Hue (màu sắc) trong HSV

**Giá trị:** Float từ 0 đến 1 (mặc định: 0.015)

**Giải thích:**

- Thay đổi màu sắc ảnh ngẫu nhiên
- Giá trị nhỏ (0.015) → Thay đổi ít, giữ nguyên màu gốc
- Giúp model không phụ thuộc vào màu sắc cụ thể

**Ví dụ:**

- `hsv_h=0.0`: Không thay đổi màu
- `hsv_h=0.015`: Thay đổi màu nhẹ
- `hsv_h=0.1`: Thay đổi màu nhiều

---

### 21. `hsv_s=0.7`

```python
hsv_s=0.7
```

**Mô tả:** Thay đổi Saturation (độ bão hòa màu)

**Giá trị:** Float từ 0 đến 1 (mặc định: 0.7)

**Giải thích:**

- Thay đổi độ đậm/nhạt của màu
- Giá trị cao (0.7) → Thay đổi nhiều
- Giúp model học với ảnh có độ bão hòa khác nhau

---

### 22. `hsv_v=0.4`

```python
hsv_v=0.4
```

**Mô tả:** Thay đổi Value (độ sáng)

**Giá trị:** Float từ 0 đến 1 (mặc định: 0.4)

**Giải thích:**

- Thay đổi độ sáng/tối của ảnh
- Giúp model học với ảnh sáng/tối khác nhau

---

### 23. `degrees=0.0`

```python
degrees=0.0
```

**Mô tả:** Xoay ảnh (rotation)

**Giá trị:** Float (mặc định: 0.0 = không xoay)

**Giải thích:**

- Xoay ảnh ngẫu nhiên trong khoảng ±degrees
- `degrees=0.0` → Không xoay (phù hợp với X-quang vì hướng quan trọng)

**Ví dụ:**

- `degrees=10.0`: Xoay ±10 độ
- `degrees=0.0`: Không xoay (đúng cho X-quang)

**Lưu ý:** Với ảnh X-quang, không nên xoay vì hướng có ý nghĩa y học

---

### 24. `translate=0.1`

```python
translate=0.1
```

**Mô tả:** Dịch chuyển ảnh (translation)

**Giá trị:** Float từ 0 đến 1 (mặc định: 0.1)

**Giải thích:**

- Dịch chuyển ảnh ngẫu nhiên
- `0.1` = Dịch chuyển tối đa 10% kích thước ảnh
- Giúp model học với vị trí object khác nhau

**Ví dụ:**

- Ảnh 512x512, `translate=0.1` → Dịch chuyển tối đa 51 pixels

---

### 25. `scale=0.5`

```python
scale=0.5
```

**Mô tả:** Thay đổi kích thước (scale)

**Giá trị:** Float (mặc định: 0.5)

**Giải thích:**

- Zoom in/out ảnh ngẫu nhiên
- `scale=0.5` → Kích thước thay đổi từ 50% đến 150%
- Giúp model học với object có kích thước khác nhau

**Ví dụ:**

- Ảnh gốc 512x512
- Scale 0.8 → 410x410
- Scale 1.2 → 614x614

---

### 26. `shear=0.0`

```python
shear=0.0
```

**Mô tả:** Biến dạng cắt (shear transformation)

**Giá trị:** Float (mặc định: 0.0 = không biến dạng)

**Giải thích:**

- Làm méo ảnh theo một hướng
- `shear=0.0` → Không biến dạng (phù hợp với X-quang)

---

### 27. `perspective=0.0`

```python
perspective=0.0
```

**Mô tả:** Biến đổi phối cảnh (perspective transformation)

**Giá trị:** Float (mặc định: 0.0 = không biến đổi)

**Giải thích:**

- Tạo hiệu ứng 3D/perspective
- `perspective=0.0` → Không biến đổi (phù hợp với X-quang phẳng)

---

### 28. `flipud=0.0`

```python
flipud=0.0
```

**Mô tả:** Lật ảnh theo chiều dọc (flip up-down)

**Giá trị:** Float từ 0 đến 1 (mặc định: 0.0 = không lật)

**Giải thích:**

- Xác suất lật ảnh theo chiều dọc
- `0.0` = 0% (không lật)
- `0.5` = 50% (lật một nửa số ảnh)

**Lưu ý:** Với X-quang, không nên lật vì hướng quan trọng

---

### 29. `fliplr=0.5`

```python
fliplr=0.5
```

**Mô tả:** Lật ảnh theo chiều ngang (flip left-right)

**Giá trị:** Float từ 0 đến 1 (mặc định: 0.5)

**Giải thích:**

- Xác suất lật ảnh theo chiều ngang
- `0.5` = 50% ảnh được lật
- Giúp tăng gấp đôi số lượng dữ liệu

**Lưu ý:** Với X-quang, có thể lật ngang vì đối xứng

---

### 30. `mosaic=1.0`

```python
mosaic=1.0
```

**Mô tả:** Ghép 4 ảnh thành 1 (Mosaic augmentation)

**Giá trị:** Float từ 0 đến 1 (mặc định: 1.0 = 100%)

**Giải thích:**

- Ghép 4 ảnh ngẫu nhiên thành 1 ảnh lớn
- Giúp model học với nhiều object cùng lúc
- Tăng hiệu quả training

**Ví dụ:**

```
[Ảnh 1] [Ảnh 2]
[Ảnh 3] [Ảnh 4]
→ Ghép thành 1 ảnh lớn
```

**Lưu ý:** Mosaic rất hiệu quả cho object detection

---

### 31. `mixup=0.0`

```python
mixup=0.0
```

**Mô tả:** Trộn 2 ảnh với nhau (Mixup augmentation)

**Giá trị:** Float từ 0 đến 1 (mặc định: 0.0 = không dùng)

**Giải thích:**

- Trộn 2 ảnh với tỷ lệ alpha
- Tạo ảnh mới = alpha * ảnh1 + (1-alpha) * ảnh2
- `mixup=0.0` → Không dùng (có thể gây nhầm lẫn cho object detection)

---

### 32. `copy_paste=0.0`

```python
copy_paste=0.0
```

**Mô tả:** Copy object từ ảnh này sang ảnh khác

**Giá trị:** Float từ 0 đến 1 (mặc định: 0.0 = không dùng)

**Giải thích:**

- Copy object từ ảnh này và paste vào ảnh khác
- Tăng số lượng object trong dataset
- `copy_paste=0.0` → Không dùng

---

## 📊 Loss Weights

Loss weights = Trọng số cho các thành phần loss khác nhau

### 33. `box=7.5`

```python
box=7.5
```

**Mô tả:** Trọng số cho Box Loss (vị trí bounding box)

**Giá trị:** Float (mặc định: 7.5)

**Giải thích:**

- Box loss = Độ lệch giữa bbox dự đoán và bbox thực tế
- Giá trị cao (7.5) → Model tập trung học vị trí bbox chính xác
- Quan trọng nhất trong object detection

**Ví dụ:**

- `box=7.5`: Trọng số cao, ưu tiên học vị trí
- `box=5.0`: Trọng số thấp hơn

---

### 34. `cls=0.5`

```python
cls=0.5
```

**Mô tả:** Trọng số cho Classification Loss (phân loại)

**Giá trị:** Float (mặc định: 0.5)

**Giải thích:**

- Classification loss = Độ lệch giữa class dự đoán và class thực tế
- Giá trị thấp (0.5) → Ít quan trọng hơn box loss
- Model đã khá tốt ở classification từ pretrained weights

---

### 35. `dfl=1.5`

```python
dfl=1.5
```

**Mô tả:** Trọng số cho Distribution Focal Loss

**Giá trị:** Float (mặc định: 1.5)

**Giải thích:**

- DFL = Loss function mới trong YOLOv8
- Giúp model học tốt hơn với object nhỏ và khó phát hiện
- Giá trị vừa phải (1.5)

---

## ✅ Validation Settings

### 36. `val=True`

```python
val=True
```

**Mô tả:** Có chạy validation không

**Giá trị:** Boolean (mặc định: `True`)

**Chức năng:**

- `True`: Sau mỗi epoch, đánh giá model trên validation set
- Tính metrics: mAP, Precision, Recall
- Lưu model tốt nhất dựa trên mAP

**Lưu ý:** Nên luôn bật để theo dõi quá trình training

---

### 37. `plots=True`

```python
plots=True
```

**Mô tả:** Có tạo plots/visualizations không

**Giá trị:** Boolean (mặc định: `True`)

**Chức năng:**

- `True`: Tạo các biểu đồ:
  - Training curves (loss, mAP)
  - Confusion matrix
  - PR curves
  - Validation predictions

**Output:** Lưu trong thư mục results

---

## 📢 Verbose Settings

### 38. `verbose=True`

```python
verbose=True
```

**Mô tả:** Có in thông tin chi tiết không

**Giá trị:** Boolean (mặc định: `True`)

**Chức năng:**

- `True`: In đầy đủ thông tin training (loss, metrics, progress)
- `False`: Chỉ in thông tin cơ bản

**Lưu ý:** Nên bật để theo dõi quá trình training

---

## 📈 Tóm tắt các tham số quan trọng

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|----------|
| `epochs` | 100 | Số lần học qua dataset |
| `batch` | 16 | Số ảnh/batch |
| `imgsz` | 512 | Kích thước ảnh |
| `lr0` | 0.01 | Learning rate ban đầu |
| `patience` | 20 | Early stopping |
| `mosaic` | 1.0 | Augmentation quan trọng |
| `fliplr` | 0.5 | Lật ngang 50% ảnh |
| `box` | 7.5 | Trọng số box loss (cao nhất) |

---

## 💡 Khuyến nghị điều chỉnh

### Nếu training chậm:

- Giảm `imgsz` (512 → 416)
- Tăng `batch` (nếu có VRAM)
- Giảm `epochs` và tăng `patience`

### Nếu overfitting:

- Tăng `weight_decay` (0.0005 → 0.001)
- Tăng data augmentation
- Tăng `patience` để early stopping sớm hơn

### Nếu không hội tụ:

- Giảm `lr0` (0.01 → 0.001)
- Tăng `warmup_epochs` (3 → 5)

---

## ✅ Kết luận

Các tham số trong `model.train()` được tối ưu cho YOLO object detection. Hầu hết giá trị mặc định đã phù hợp, chỉ cần điều chỉnh:

- `epochs`, `batch`, `imgsz` theo tài nguyên
- `lr0` nếu training không ổn định
- Data augmentation theo đặc thù dataset (X-quang không nên xoay/lật dọc)

