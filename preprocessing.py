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
        
        # Class mapping - cập nhật theo format thực tế
        self.class_map = {
            'healthy': 0,
            'sick_but_no_tb': 1, 
            'tb': 2
        }
        
        # Target mapping
        self.target_map = {
            'no_tb': 0,    # healthy hoặc sick_but_no_tb
            'tb': 2        # tb
        }
        
    def parse_csv(self):
        """Đọc và parse CSV file với format đúng"""
        print("📖 Đọc file CSV...")
        
        # Đọc CSV với header đúng theo format thực tế
        self.df = pd.read_csv(self.csv_path)
        
        # In ra để kiểm tra
        print(f"✅ Đã load {len(self.df)} samples")
        print(f"📊 Columns: {list(self.df.columns)}")
        print(f"📊 Phân bố target: {self.df['target'].value_counts().to_dict()}")
        print(f"📊 Phân bố image_type: {self.df['image_type'].value_counts().to_dict()}")
        print(f"📊 Phân bố source: {self.df['source'].value_counts().to_dict()}")
        
        return self.df
    
    def create_directories(self):
        """Tạo cấu trúc thư mục YOLO"""
        print("\n📁 Tạo cấu trúc thư mục...")
        
        splits = ['train', 'val']
        
        for split in splits:
            os.makedirs(f'{self.output_dir}/images/{split}', exist_ok=True)
            os.makedirs(f'{self.output_dir}/labels/{split}', exist_ok=True)
        
        print("✅ Đã tạo thư mục")
    
    def find_image_path(self, filename):
        """Tìm đường dẫn thực tế của ảnh"""
        # Các nơi có thể chứa ảnh
        possible_paths = [
            os.path.join(self.images_dir, filename),
            os.path.join(self.images_dir, 'train', filename),
            os.path.join(self.images_dir, 'val', filename),
            os.path.join(self.images_dir, 'test', filename),
            # Thêm test directory riêng
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
        """Parse bbox string an toàn với error handling"""
        try:
            if pd.isna(bbox_str) or bbox_str == 'none' or bbox_str == '':
                return None
                
            # Nếu là string, thử parse
            if isinstance(bbox_str, str):
                # Thử parse như dictionary
                bbox_dict = ast.literal_eval(bbox_str)
                return bbox_dict
            else:
                return bbox_str
                
        except Exception as e:
            print(f"⚠️ Lỗi parse bbox: {bbox_str} - {e}")
            return None
    
    def convert_to_yolo_format(self):
        """Chuyển đổi bbox sang YOLO format"""
        print("\n🔄 Chuyển đổi dataset sang YOLO format...")
        
        converted_count = 0
        skipped_count = 0
        missing_images = []
        
        for idx, row in self.df.iterrows():
            try:
                filename = row['fname']  # Sử dụng 'fname' thay vì 'filename'
                split = row['source']    # Sử dụng 'source' thay vì 'split'
                
                # Tìm ảnh
                src_image = self.find_image_path(filename)
                dst_image = os.path.join(self.output_dir, 'images', split, filename)
                
                if src_image is not None and os.path.exists(src_image):
                    shutil.copy2(src_image, dst_image)
                else:
                    if skipped_count < 10:  # Chỉ in 10 lỗi đầu
                        print(f"⚠️  Không tìm thấy ảnh: {filename}")
                    missing_images.append(filename)
                    skipped_count += 1
                    continue
                
                # Tạo file label
                label_filename = filename.replace('.png', '.txt')
                label_path = os.path.join(self.output_dir, 'labels', split, label_filename)
                
                # Parse bounding box - FIX ĐÂY LÀ NGUYÊN NHÂN LỖI
                bbox_data = self.safe_parse_bbox(row['bbox'])
                
                if bbox_data is not None:
                    try:
                        # Đảm bảo các giá trị là số
                        img_width = float(row['image_width'])
                        img_height = float(row['image_height'])
                        
                        # Parse bbox coordinates
                        if isinstance(bbox_data, dict):
                            # Format: {'xmin': x, 'ymin': y, 'width': w, 'height': h}
                            xmin = float(bbox_data['xmin'])
                            ymin = float(bbox_data['ymin'])
                            bbox_width = float(bbox_data['width'])
                            bbox_height = float(bbox_data['height'])
                        else:
                            print(f"⚠️ Unexpected bbox format: {bbox_data}")
                            continue
                        
                        # Chuyển sang YOLO format (normalized)
                        x_center = (xmin + bbox_width/2) / img_width
                        y_center = (ymin + bbox_height/2) / img_height
                        width = bbox_width / img_width
                        height = bbox_height / img_height
                        
                        # Đảm bảo giá trị trong khoảng [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        # Map class dựa trên target và image_type
                        if row['target'] == 'tb':
                            class_id = 2  # tb
                        elif row['image_type'] == 'healthy':
                            class_id = 0  # healthy
                        else:
                            class_id = 1  # sick_but_no_tb
                        
                        # Ghi vào file (append nếu có nhiều bbox)
                        with open(label_path, 'a') as f:
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        
                        converted_count += 1
                        
                    except Exception as e:
                        print(f"⚠️  Lỗi parse bbox {filename}: {e}")
                        print(f"   Row data: {row.to_dict()}")
                        skipped_count += 1
                else:
                    # Tạo file label rỗng cho ảnh healthy/sick_but_no_tb
                    with open(label_path, 'w') as f:
                        pass
                
                # Progress
                if (idx + 1) % 100 == 0:
                    print(f"Đã xử lý: {idx + 1}/{len(self.df)}")
                    
            except Exception as e:
                print(f"❌ Lỗi xử lý {row.get('fname', 'unknown')}: {e}")
                skipped_count += 1
        
        print(f"\n✅ Hoàn tất!")
        print(f"   - Converted: {converted_count}")
        print(f"   - Skipped: {skipped_count}")
        
        # Lưu danh sách ảnh bị thiếu
        if missing_images:
            missing_file = os.path.join(self.output_dir, 'missing_images.txt')
            with open(missing_file, 'w') as f:
                for img in missing_images:
                    f.write(f"{img}\n")
            print(f"   - Missing images saved to: {missing_file}")
            print(f"\n⚠️  CÓ {len(missing_images)} ẢNH THIẾU!")
            print("   Kiểm tra lại cấu trúc thư mục images/")
    
    def create_yaml_config(self):
        """Tạo file dataset.yaml cho YOLO"""
        print("\n📝 Tạo file config YAML...")
        
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
        
        print(f"✅ Đã tạo: {yaml_path}")
        return yaml_path
    
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
            print(self.df[self.df['target'] == 'tb']['tb_type'].value_counts())
        
        # Số lượng có bbox
        has_bbox = self.df['bbox'].apply(lambda x: x != 'none' and pd.notna(x)).sum()
        print(f"\n5. Số ảnh có bounding box: {has_bbox}/{len(self.df)}")
    
    def run(self):
        """Chạy toàn bộ preprocessing pipeline"""
        self.parse_csv()
        self.analyze_dataset()
        self.create_directories()
        self.convert_to_yolo_format()
        yaml_path = self.create_yaml_config()
        
        print("\n" + "="*50)
        print("✅ PREPROCESSING HOÀN TẤT!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📝 Config file: {yaml_path}")
        print("="*50)
        
        return yaml_path