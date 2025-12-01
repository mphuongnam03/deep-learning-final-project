import pandas as pd
import os
import ast
import shutil
from pathlib import Path
import cv2
import numpy as np

class TBDataPreprocessor:
    def __init__(self, csv_path, images_dir, output_dir):
        """
        Args:
            csv_path: Đường dẫn đến data.csv
            images_dir: Thư mục chứa ảnh (images/)
            output_dir: Thư mục output cho YOLO format
        """
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        
        # Ánh xạ lớp - 4 lớp
        self.class_map = {
            'healthy': 0,            # Khỏe mạnh
            'sick_but_no_tb': 1,     # Bệnh nhưng không phải TB
            'active_tb': 2,          # Lao hoạt động
            'latent_tb': 3           # Lao tiềm ẩn
        }
        
        # Ánh xạ mục tiêu (giữ lại để tương thích ngược)
        self.target_map = {
            'no_tb': 0,    # healthy hoặc sick_but_no_tb
            'tb': 2        # tb (sẽ được phân chia thành active/latent)
        }
        
    def parse_csv(self):
        """Đọc và phân tích file CSV với định dạng chuẩn"""
        print("📖 Đang đọc file CSV...")
        
        # Đọc CSV với header đúng theo định dạng thực tế
        self.df = pd.read_csv(self.csv_path)
        
        # In ra để kiểm tra
        print(f"✅ Đã tải {len(self.df)} mẫu")
        print(f"📊 Các cột: {list(self.df.columns)}")
        print(f"📊 Phân bố target: {self.df['target'].value_counts().to_dict()}")
        print(f"📊 Phân bố image_type: {self.df['image_type'].value_counts().to_dict()}")
        print(f"📊 Phân bố source: {self.df['source'].value_counts().to_dict()}")
        
        return self.df
    
    def create_directories(self):
        """Tạo cấu trúc thư mục cho YOLO"""
        print("\n📁 Đang tạo cấu trúc thư mục...")
        
        splits = ['train', 'val']
        
        for split in splits:
            os.makedirs(f'{self.output_dir}/images/{split}', exist_ok=True)
            os.makedirs(f'{self.output_dir}/labels/{split}', exist_ok=True)
        
        print("✅ Đã tạo xong thư mục")
    
    def find_image_path(self, filename):
        """Tìm đường dẫn thực tế của ảnh"""
        # Các vị trí có thể chứa ảnh
        possible_paths = [
            os.path.join(self.images_dir, filename),
            os.path.join(self.images_dir, 'train', filename),
            os.path.join(self.images_dir, 'val', filename),
            os.path.join(self.images_dir, 'test', filename),
            # Thêm thư mục test riêng
            os.path.join(os.path.dirname(self.images_dir), 'test', filename),
            filename,  # Đường dẫn trực tiếp
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Tìm kiếm đệ quy
        for root, dirs, files in os.walk(self.images_dir):
            if filename in files:
                return os.path.join(root, filename)
        
        return None
    
    def safe_parse_bbox(self, bbox_str):
        """Phân tích chuỗi bbox an toàn với xử lý lỗi"""
        try:
            if pd.isna(bbox_str) or bbox_str == 'none' or bbox_str == '':
                return None
                
            # Nếu là string, thử phân tích
            if isinstance(bbox_str, str):
                # Thử phân tích như dictionary
                bbox_dict = ast.literal_eval(bbox_str)
                return bbox_dict
            else:
                return bbox_str
                
        except Exception as e:
            print(f"⚠️ Lỗi parse bbox: {bbox_str} - {e}")
            return None
    
    def convert_to_yolo_format(self):
        """
        Chuyển đổi bbox sang định dạng YOLO cho Phát hiện.
        Chỉ tạo nhãn cho các ảnh TB có bounding box (2 lớp: active_tb, latent_tb).
        """
        print("\n🔄 Đang chuyển đổi dataset sang định dạng YOLO Detection...")
        print("   (Chỉ các ảnh TB có bounding box)")
        
        converted_count = 0
        skipped_count = 0
        missing_images = []
        
        # Ánh xạ lớp Detection (chỉ 2 lớp)
        detection_class_map = {
            'active_tb': 0,    # Lao hoạt động
            'latent_tb': 1     # Lao tiềm ẩn
        }
        
        for idx, row in self.df.iterrows():
            try:
                filename = row['fname']
                split = row['source']
                
                # Chỉ xử lý ảnh TB có bounding box
                if row['target'] != 'tb':
                    continue
                    
                bbox_data = self.safe_parse_bbox(row['bbox'])
                if bbox_data is None:
                    continue
                
                # Tìm và sao chép ảnh
                src_image = self.find_image_path(filename)
                dst_image = os.path.join(self.output_dir, 'images', split, filename)
                
                if src_image is not None and os.path.exists(src_image):
                    shutil.copy2(src_image, dst_image)
                else:
                    if skipped_count < 10:
                        print(f"⚠️  Không tìm thấy ảnh: {filename}")
                    missing_images.append(filename)
                    skipped_count += 1
                    continue
                
                # Tạo file nhãn
                label_filename = filename.replace('.png', '.txt').replace('.jpg', '.txt')
                label_path = os.path.join(self.output_dir, 'labels', split, label_filename)
                
                try:
                    img_width = float(row['image_width'])
                    img_height = float(row['image_height'])
                    
                    if isinstance(bbox_data, dict):
                        xmin = float(bbox_data['xmin'])
                        ymin = float(bbox_data['ymin'])
                        bbox_width = float(bbox_data['width'])
                        bbox_height = float(bbox_data['height'])
                    else:
                        print(f"⚠️ Định dạng bbox không mong đợi: {bbox_data}")
                        continue
                    
                    # Chuyển sang định dạng YOLO (chuẩn hóa)
                    x_center = (xmin + bbox_width/2) / img_width
                    y_center = (ymin + bbox_height/2) / img_height
                    width = bbox_width / img_width
                    height = bbox_height / img_height
                    
                    # Giới hạn giá trị trong [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # Ánh xạ sang lớp detection (0 = active_tb, 1 = latent_tb)
                    tb_type = row.get('tb_type', 'active_tb')
                    if pd.notna(tb_type) and tb_type == 'latent_tb':
                        class_id = 1  # lao tiềm ẩn
                    else:
                        class_id = 0  # lao hoạt động (mặc định)
                    
                    # Ghi vào file nhãn (nối thêm cho nhiều bbox)
                    with open(label_path, 'a') as f:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    converted_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Lỗi phân tích bbox {filename}: {e}")
                    skipped_count += 1
                
                # Tiến độ
                if (idx + 1) % 100 == 0:
                    print(f"   Đã xử lý: {idx + 1}/{len(self.df)}")
                    
            except Exception as e:
                print(f"❌ Lỗi xử lý {row.get('fname', 'unknown')}: {e}")
                skipped_count += 1
        
        print(f"\n✅ Đã tạo dataset Phát hiện!")
        print(f"   - Đã chuyển đổi: {converted_count} ảnh TB có bbox")
        print(f"   - Bỏ qua: {skipped_count}")
        
        if missing_images:
            missing_file = os.path.join(self.output_dir, 'missing_images.txt')
            with open(missing_file, 'w') as f:
                for img in missing_images:
                    f.write(f"{img}\n")
            print(f"   - Danh sách ảnh thiếu lưu tại: {missing_file}")
    
    def create_yaml_config(self):
        """Tạo file dataset.yaml cho YOLO Detection"""
        print("\n📝 Đang tạo file cấu hình YAML...")
        
        yaml_content = f"""# Cấu hình Dataset TBX11K - Phát hiện
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val

# Các lớp - 2 lớp cho phát hiện (chỉ tổn thương TB)
nc: 2
names: ['active_tb', 'latent_tb']

# Thông tin dataset
roboflow:
  workspace: tbx11k
  project: tb-detection
  version: 1
"""
        
        yaml_path = os.path.join(self.output_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Đã tạo: {yaml_path}")
        return yaml_path
    
    def create_classification_dataset(self):
        """
        Tạo dataset cho YOLOv8-cls (Phân loại).
        Cấu trúc: output_dir/classification/train|val/tên_lớp/các_ảnh
        """
        print("\n📁 Đang tạo dataset cho Phân loại (4 lớp)...")
        
        cls_output = os.path.join(self.output_dir, 'classification')
        splits = ['train', 'val']
        class_names = ['healthy', 'sick_but_no_tb', 'active_tb', 'latent_tb']
        
        # Tạo cấu trúc thư mục
        for split in splits:
            for class_name in class_names:
                os.makedirs(os.path.join(cls_output, split, class_name), exist_ok=True)
        
        copied_count = {split: {cn: 0 for cn in class_names} for split in splits}
        skipped = 0
        
        for idx, row in self.df.iterrows():
            try:
                filename = row['fname']
                split = row['source']  # 'train' hoặc 'val'
                
                # Xác định lớp dựa trên image_type và tb_type
                if row['image_type'] == 'healthy':
                    class_name = 'healthy'
                elif row['image_type'] == 'sick_but_no_tb':
                    class_name = 'sick_but_no_tb'
                elif row['target'] == 'tb':
                    if pd.notna(row.get('tb_type')) and row['tb_type'] == 'latent_tb':
                        class_name = 'latent_tb'
                    else:
                        class_name = 'active_tb'  # Mặc định cho TB không có loại
                else:
                    class_name = 'healthy'  # Dự phòng
                
                # Tìm và sao chép ảnh
                src_image = self.find_image_path(filename)
                if src_image and os.path.exists(src_image):
                    dst_path = os.path.join(cls_output, split, class_name, filename)
                    shutil.copy2(src_image, dst_path)
                    copied_count[split][class_name] += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                skipped += 1
        
        # In tóm tắt
        print("\n📊 Đã tạo dataset Phân loại:")
        for split in splits:
            print(f"\n   {split.upper()}:")
            for class_name in class_names:
                print(f"      {class_name}: {copied_count[split][class_name]}")
        print(f"\n   Bỏ qua: {skipped}")
        
        return cls_output
    
    def analyze_dataset(self):
        """Phân tích thống kê dataset"""
        print("\n📊 PHÂN TÍCH DATASET:")
        print("="*50)
        
        # Phân bố theo source (train/val)
        print("\n1. Phân bố theo source:")
        print(self.df['source'].value_counts())
        
        # Phân bố theo target
        print("\n2. Phân bố theo target:")
        print(self.df['target'].value_counts())
        
        # Phân bố theo image_type
        print("\n3. Phân bố theo image_type:")
        print(self.df['image_type'].value_counts())
        
        # TB type distribution
        if 'tb_type' in self.df.columns:
            print("\n4. TB type distribution:")
            tb_type_dist = self.df[self.df['target'] == 'tb']['tb_type'].value_counts()
            print(tb_type_dist)
        
        # Phân bố 4 classes mới
        print("\n5. Phân bố 4 classes:")
        healthy_count = len(self.df[self.df['image_type'] == 'healthy'])
        sick_count = len(self.df[self.df['image_type'] == 'sick_but_no_tb'])
        active_tb_count = len(self.df[(self.df['target'] == 'tb') & 
                                     (self.df['tb_type'] == 'active_tb')])
        latent_tb_count = len(self.df[(self.df['target'] == 'tb') & 
                                      (self.df['tb_type'] == 'latent_tb')])
        tb_no_type = len(self.df[(self.df['target'] == 'tb') & 
                                 (self.df['tb_type'].isna() | 
                                  (self.df['tb_type'] == 'none'))])
        
        total = len(self.df)
        class_dist = {
            'healthy': healthy_count,
            'sick_but_no_tb': sick_count,
            'active_tb': active_tb_count,
            'latent_tb': latent_tb_count
        }
        
        for class_name, count in class_dist.items():
            percentage = count / total * 100 if total > 0 else 0
            print(f"   {class_name:15s}: {count:5d} ({percentage:5.1f}%)")
        
        if tb_no_type > 0:
            print(f"   {'tb (no type)':15s}: {tb_no_type:5d} ({tb_no_type/total*100:5.1f}%)")
            print(f"   ⚠️  {tb_no_type} ảnh TB không có tb_type sẽ được gán vào active_tb")
        
        # Tính tỷ lệ mất cân bằng
        if len(class_dist) > 0:
            max_count = max(class_dist.values())
            min_count = min([v for v in class_dist.values() if v > 0])
            if min_count > 0:
                imbalance_ratio = max_count / min_count
                print(f"   Tỷ lệ mất cân bằng: {imbalance_ratio:.2f}:1")
                if imbalance_ratio < 2.0:
                    print("   ✅ Cân bằng tốt")
                elif imbalance_ratio < 3.0:
                    print("   ⚠️  Cân bằng chấp nhận được")
                else:
                    print("   ❌ Mất cân bằng nghiêm trọng - nên cân bằng lại dataset")
        
        # Số lượng có bbox
        has_bbox = self.df['bbox'].apply(lambda x: x != 'none' and pd.notna(x)).sum()
        print(f"\n6. Số ảnh có bounding box: {has_bbox}/{len(self.df)}")
    
    def run(self):
        """Chạy toàn bộ pipeline tiền xử lý"""
        self.parse_csv()
        self.analyze_dataset()
        self.create_directories()
        self.convert_to_yolo_format()
        yaml_path = self.create_yaml_config()
        
        # Create classification dataset for YOLOv8-cls
        cls_path = self.create_classification_dataset()
        
        print("\n" + "="*50)
        print("✅ TIỀN XỬ LÝ HOÀN TẤT!")
        print("="*50)
        print(f"\n📁 Dataset PHÁT HIỆN:")
        print(f"   Đường dẫn: {self.output_dir}")
        print(f"   Cấu hình: {yaml_path}")
        print(f"   Các lớp: ['active_tb', 'latent_tb']")
        print(f"\n📁 Dataset PHÂN LOẠI:")
        print(f"   Đường dẫn: {cls_path}")
        print(f"   Các lớp: ['healthy', 'sick_but_no_tb', 'active_tb', 'latent_tb']")
        print("\n" + "="*50)
        
        return yaml_path