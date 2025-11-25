# Giải thích chi tiết file main.py

## Tổng quan

File `main.py` là entry point chính của project, điều phối toàn bộ pipeline từ preprocessing, training, evaluation đến visualization. File này sử dụng argparse để nhận tham số từ command line và gọi các module tương ứng.

---

## Giải thích từng dòng

### Dòng 1-3: Docstring và import

```python
"""
Main script để chạy toàn bộ pipeline
"""
```

**Giải thích**: Docstring mô tả mục đích của file - đây là script chính để chạy toàn bộ pipeline.

---

### Dòng 5: Import argparse

```python
import argparse
```

**Giải thích**: Import module `argparse` để xử lý các tham số dòng lệnh (command-line arguments). Module này cho phép định nghĩa và parse các tham số như `--mode`, `--csv`, `--images`, etc.

---

### Dòng 6-9: Import các module của project

```python
from preprocessing import TBDataPreprocessor
from train import TBModelTrainer
from evaluate import TBModelEvaluator
from heatmap import TBHeatmapGenerator
```

**Giải thích**: 
- Import các class chính từ các module khác trong project:
  - `TBDataPreprocessor`: Xử lý và chuyển đổi dữ liệu
  - `TBModelTrainer`: Huấn luyện model YOLO
  - `TBModelEvaluator`: Đánh giá model
  - `TBHeatmapGenerator`: Tạo heatmap visualization

---

### Dòng 11: Định nghĩa hàm main()

```python
def main():
```

**Giải thích**: Định nghĩa hàm `main()` - hàm chính của script. Tất cả logic chính sẽ được thực thi trong hàm này.

---

### Dòng 12: Tạo ArgumentParser

```python
    parser = argparse.ArgumentParser(description='TB Detection Training Pipeline')
```

**Giải thích**: 
- Tạo một `ArgumentParser` object để xử lý các tham số dòng lệnh
- `description`: Mô tả ngắn gọn về script, sẽ hiển thị khi chạy `python main.py --help`

---

### Dòng 13-15: Tham số --mode

```python
    parser.add_argument('--mode', type=str, required=True,
                       choices=['preprocess', 'train', 'evaluate', 'heatmap', 'all'],
                       help='Chế độ chạy')
```

**Giải thích**: 
- Định nghĩa tham số `--mode` (bắt buộc)
- `type=str`: Kiểu dữ liệu là string
- `required=True`: Tham số bắt buộc phải có
- `choices=[...]`: Chỉ chấp nhận các giá trị trong danh sách:
  - `preprocess`: Chỉ chạy preprocessing
  - `train`: Chỉ chạy training
  - `evaluate`: Chỉ chạy evaluation
  - `heatmap`: Chỉ tạo heatmap
  - `all`: Chạy tất cả các bước
- `help`: Mô tả hiển thị khi dùng `--help`

---

### Dòng 16-17: Tham số --csv

```python
    parser.add_argument('--csv', type=str, default='data.csv',
                       help='Đường dẫn đến data.csv')
```

**Giải thích**: 
- Định nghĩa tham số `--csv` (tùy chọn)
- `default='data.csv'`: Giá trị mặc định nếu không chỉ định
- Dùng để chỉ định đường dẫn đến file CSV chứa metadata của dataset

---

### Dòng 18-19: Tham số --images

```python
    parser.add_argument('--images', type=str, default='images',
                       help='Thư mục chứa ảnh')
```

**Giải thích**: 
- Định nghĩa tham số `--images` (tùy chọn)
- `default='images'`: Thư mục mặc định chứa ảnh
- Dùng để chỉ định thư mục chứa các file ảnh X-quang

---

### Dòng 20-21: Tham số --output

```python
    parser.add_argument('--output', type=str, default='tbx11k-simplified',
                       help='Thư mục output')
```

**Giải thích**: 
- Định nghĩa tham số `--output` (tùy chọn)
- `default='tbx11k-simplified'`: Thư mục output mặc định
- Dùng để chỉ định thư mục lưu kết quả sau khi preprocessing

---

### Dòng 22-23: Tham số --model

```python
    parser.add_argument('--model', type=str, default='best.pt',
                       help='Đường dẫn đến model')
```

**Giải thích**: 
- Định nghĩa tham số `--model` (tùy chọn)
- `default='best.pt'`: File model mặc định
- Dùng để chỉ định đường dẫn đến file model (.pt) khi evaluate hoặc tạo heatmap

---

### Dòng 24-25: Tham số --epochs

```python
    parser.add_argument('--epochs', type=int, default=100,
                       help='Số epochs')
```

**Giải thích**: 
- Định nghĩa tham số `--epochs` (tùy chọn)
- `type=int`: Kiểu dữ liệu là số nguyên
- `default=100`: Số epochs mặc định là 100
- Dùng để chỉ định số lần model sẽ được huấn luyện qua toàn bộ dataset

---

### Dòng 26-27: Tham số --batch

```python
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
```

**Giải thích**: 
- Định nghĩa tham số `--batch` (tùy chọn)
- `type=int`: Kiểu dữ liệu là số nguyên
- `default=16`: Batch size mặc định là 16
- Dùng để chỉ định số lượng ảnh được xử lý cùng lúc trong mỗi batch khi training

---

### Dòng 28-29: Tham số --img-size

```python
    parser.add_argument('--img-size', type=int, default=512,
                       help='Image size')
```

**Giải thích**: 
- Định nghĩa tham số `--img-size` (tùy chọn)
- `type=int`: Kiểu dữ liệu là số nguyên
- `default=512`: Kích thước ảnh mặc định là 512x512 pixels
- Dùng để chỉ định kích thước ảnh đầu vào cho model (ảnh sẽ được resize về kích thước này)

---

### Dòng 31: Parse arguments

```python
    args = parser.parse_args()
```

**Giải thích**: 
- Parse tất cả các tham số từ command line
- Kết quả được lưu vào object `args`
- Có thể truy cập các giá trị qua `args.mode`, `args.csv`, `args.images`, etc.

---

### Dòng 33-36: Xử lý mode preprocess hoặc all

```python
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n🔄 BƯỚC 1: PREPROCESSING")
        preprocessor = TBDataPreprocessor(args.csv, args.images, args.output)
        yaml_path = preprocessor.run()
```

**Giải thích**: 
- Kiểm tra nếu mode là `preprocess` hoặc `all`
- In thông báo bắt đầu bước preprocessing
- Tạo instance của `TBDataPreprocessor` với 3 tham số:
  - `args.csv`: Đường dẫn file CSV
  - `args.images`: Thư mục chứa ảnh
  - `args.output`: Thư mục output
- Gọi `preprocessor.run()` để chạy toàn bộ pipeline preprocessing:
  - Đọc CSV
  - Phân tích dataset
  - Chuyển đổi sang YOLO format
  - Tạo dataset.yaml
- Lưu đường dẫn file YAML vào biến `yaml_path` (nhưng không dùng ở đây)

---

### Dòng 38-45: Xử lý mode train hoặc all

```python
    if args.mode == 'train' or args.mode == 'all':
        print("\n🔄 BƯỚC 2: TRAINING")
        yaml_path = f"{args.output}/dataset.yaml"
        trainer = TBModelTrainer(yaml_path, model_size='n')
        results = trainer.train(epochs=args.epochs, batch_size=args.batch, 
                               img_size=args.img_size)
        
        print(f"\n✅ Model saved to: {results.save_dir}/weights/best.pt")
```

**Giải thích**: 
- Kiểm tra nếu mode là `train` hoặc `all`
- In thông báo bắt đầu bước training
- Tạo đường dẫn đến file `dataset.yaml` từ thư mục output
- Tạo instance của `TBModelTrainer` với:
  - `yaml_path`: Đường dẫn file dataset.yaml
  - `model_size='n'`: Sử dụng YOLOv8 nano (model nhỏ nhất, nhanh nhất)
- Gọi `trainer.train()` với các tham số:
  - `epochs=args.epochs`: Số epochs từ command line
  - `batch_size=args.batch`: Batch size từ command line
  - `img_size=args.img_size`: Kích thước ảnh từ command line
- Kết quả training được lưu vào `results`
- In đường dẫn đến file model tốt nhất (`best.pt`)

---

### Dòng 47-51: Xử lý mode evaluate

```python
    if args.mode == 'evaluate':
        print("\n🔄 EVALUATION")
        yaml_path = f"{args.output}/dataset.yaml"
        evaluator = TBModelEvaluator(args.model)
        evaluator.evaluate_on_validation(yaml_path)
```

**Giải thích**: 
- Kiểm tra nếu mode là `evaluate` (chỉ chạy evaluation, không chạy các bước khác)
- In thông báo bắt đầu bước evaluation
- Tạo đường dẫn đến file `dataset.yaml`
- Tạo instance của `TBModelEvaluator` với:
  - `args.model`: Đường dẫn đến file model (.pt)
- Gọi `evaluator.evaluate_on_validation()` để đánh giá model trên validation set:
  - Tính các metrics: mAP50, mAP50-95, Precision, Recall
  - Hiển thị kết quả

---

### Dòng 53-57: Xử lý mode heatmap

```python
    if args.mode == 'heatmap':
        print("\n🔄 HEATMAP GENERATION")
        generator = TBHeatmapGenerator(args.model)
        # Ví dụ: tạo heatmap cho ảnh test
        generator.generate_batch_heatmaps('test', 'test_heatmaps')
```

**Giải thích**: 
- Kiểm tra nếu mode là `heatmap` (chỉ tạo heatmap, không chạy các bước khác)
- In thông báo bắt đầu tạo heatmap
- Tạo instance của `TBHeatmapGenerator` với:
  - `args.model`: Đường dẫn đến file model (.pt)
- Gọi `generate_batch_heatmaps()` để tạo heatmap cho nhiều ảnh:
  - Tham số 1: `'test'` - thư mục chứa ảnh test (hardcoded, có thể cải thiện)
  - Tham số 2: `'test_heatmaps'` - thư mục lưu kết quả heatmap

**Lưu ý**: Đường dẫn `'test'` và `'test_heatmaps'` đang được hardcode, nên có thể không hoạt động nếu thư mục không tồn tại.

---

### Dòng 59-60: Entry point

```python
if __name__ == '__main__':
    main()
```

**Giải thích**: 
- `if __name__ == '__main__'`: Kiểm tra nếu file được chạy trực tiếp (không phải import)
- Chỉ khi file được chạy trực tiếp bằng `python main.py`, hàm `main()` mới được gọi
- Nếu file được import từ module khác, đoạn code này sẽ không chạy

---

## Ví dụ sử dụng

### Chạy toàn bộ pipeline:
```bash
python main.py --mode all \
    --csv tbx11k-simplified/data.csv \
    --images tbx11k-simplified/images \
    --output tbx11k-simplified \
    --epochs 100 \
    --batch 16 \
    --img-size 512
```

### Chỉ chạy preprocessing:
```bash
python main.py --mode preprocess \
    --csv data.csv \
    --images images/ \
    --output output/
```

### Chỉ chạy training:
```bash
python main.py --mode train \
    --output tbx11k-simplified \
    --epochs 50 \
    --batch 32
```

### Chỉ chạy evaluation:
```bash
python main.py --mode evaluate \
    --model best.pt \
    --output tbx11k-simplified
```

---

## Lưu ý

1. **Thứ tự thực thi**: Khi dùng `--mode all`, các bước sẽ chạy theo thứ tự: preprocess → train. Evaluation và heatmap không chạy trong mode `all`.

2. **Thiếu evaluation trong mode all**: Mode `all` chỉ chạy preprocess và train, không chạy evaluate và heatmap. Cần chạy riêng các mode này.

3. **Hardcoded paths**: Trong mode heatmap, đường dẫn `'test'` và `'test_heatmaps'` đang được hardcode, nên có thể cần chỉnh sửa.

4. **Biến yaml_path**: Biến `yaml_path` được tạo trong mode preprocess nhưng không được sử dụng trong các mode khác (mỗi mode tự tạo lại).

---

## Cải thiện có thể

1. Thêm tham số `--test-dir` và `--heatmap-output` cho mode heatmap
2. Thêm mode `all` để chạy cả evaluation sau training
3. Sử dụng biến `yaml_path` từ preprocess thay vì tạo lại
4. Thêm error handling cho các trường hợp lỗi
5. Thêm logging thay vì chỉ dùng print

