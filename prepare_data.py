"""
Chuẩn bị dữ liệu cho YOLOv8 Classification và Detection
=========================================================
Script này xử lý dữ liệu X-quang ngực thô thành 2 dataset riêng biệt:
1. dataset_cls: Cho YOLOv8-cls (4 lớp phân loại)
2. dataset_det: Cho YOLOv8-det (2 lớp phát hiện + background images)

Tác giả: AI4Life Team
Ngày: 2025
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import random
import yaml

# Seed cho reproducibility
random.seed(42)
np.random.seed(42)


class DataPreparer:
    """
    Lớp xử lý và chuẩn bị dữ liệu cho YOLOv8.
    
    Attributes:
        csv_path: Đường dẫn đến file CSV chứa metadata
        images_dir: Thư mục chứa ảnh gốc
        output_dir: Thư mục output chính
        val_ratio: Tỷ lệ validation (mặc định 0.2 = 20%)
    """
    
    def __init__(self, csv_path: str, images_dir: str, output_dir: str, val_ratio: float = 0.2):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.val_ratio = val_ratio
        
        # Thư mục output cho 2 dataset
        self.cls_dir = os.path.join(output_dir, 'dataset_cls')
        self.det_dir = os.path.join(output_dir, 'dataset_det')
        
        # Mapping lớp
        self.cls_classes = ['healthy', 'sick_but_no_tb', 'active_tb', 'latent_tb']
        self.det_classes = ['active_tb', 'latent_tb']  # Chỉ 2 lớp cho detection
        
        # Load data
        self.df = None
        
    def load_data(self):
        """Đọc và phân tích file CSV"""
        print("📖 Đang đọc file CSV...")
        self.df = pd.read_csv(self.csv_path)
        
        print(f"✅ Đã tải {len(self.df)} mẫu")
        print(f"\n📊 Phân bố dữ liệu ban đầu:")
        
        # Phân loại ảnh theo class
        self.df['class_name'] = self.df.apply(self._get_class_name, axis=1)
        
        class_counts = self.df['class_name'].value_counts()
        for cls, count in class_counts.items():
            print(f"   {cls}: {count} ảnh")
            
        return self.df
    
    def _get_class_name(self, row):
        """Xác định tên lớp từ row data"""
        if row['image_type'] == 'healthy':
            return 'healthy'
        elif row['image_type'] == 'sick_but_no_tb':
            return 'sick_but_no_tb'
        elif row['target'] == 'tb':
            tb_type = row.get('tb_type', 'active_tb')
            if pd.notna(tb_type) and tb_type == 'latent_tb':
                return 'latent_tb'
            else:
                return 'active_tb'
        return 'healthy'  # Fallback
    
    def find_image_path(self, filename: str) -> str:
        """Tìm đường dẫn thực tế của ảnh"""
        possible_paths = [
            os.path.join(self.images_dir, filename),
            os.path.join(self.images_dir, 'train', filename),
            os.path.join(self.images_dir, 'val', filename),
            os.path.join(self.images_dir, 'test', filename),
            os.path.join(os.path.dirname(self.images_dir), 'test', filename),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Tìm kiếm đệ quy
        base_dir = Path(self.images_dir).parent
        for root, dirs, files in os.walk(base_dir):
            if filename in files:
                return os.path.join(root, filename)
        
        return None
    
    # =========================================================================
    # OFFLINE AUGMENTATION CHO LỚP THIỂU SỐ (latent_tb)
    # =========================================================================
    
    def offline_augmentation(self, target_class: str = 'latent_tb', target_count: int = 800):
        """
        Tăng cường dữ liệu OFFLINE cho lớp thiểu số.
        
        Ý tưởng: Vì latent_tb chỉ có 239 ảnh (rất ít so với các lớp khác),
        ta cần tạo thêm ảnh augmented TRƯỚC KHI chia train/val để:
        - Cân bằng dataset
        - Giúp model học được nhiều biến thể hơn của lớp này
        
        Args:
            target_class: Lớp cần augment (mặc định: latent_tb)
            target_count: Số lượng ảnh mục tiêu sau augmentation
        """
        print(f"\n🔄 Đang thực hiện Offline Augmentation cho lớp '{target_class}'...")
        
        # Lọc ảnh của lớp cần augment
        class_df = self.df[self.df['class_name'] == target_class].copy()
        current_count = len(class_df)
        
        if current_count >= target_count:
            print(f"   ℹ️  Lớp {target_class} đã có {current_count} ảnh, không cần augment")
            return
        
        # Số ảnh cần tạo thêm
        needed = target_count - current_count
        print(f"   📈 Cần tạo thêm: {needed} ảnh (hiện có: {current_count}, mục tiêu: {target_count})")
        
        # Tạo thư mục tạm cho ảnh augmented
        aug_dir = os.path.join(self.output_dir, 'augmented_temp', target_class)
        os.makedirs(aug_dir, exist_ok=True)
        
        # Danh sách ảnh mới
        new_rows = []
        aug_count = 0
        
        while aug_count < needed:
            for idx, row in class_df.iterrows():
                if aug_count >= needed:
                    break
                    
                src_path = self.find_image_path(row['fname'])
                if src_path is None:
                    continue
                
                # Đọc ảnh
                img = cv2.imread(src_path)
                if img is None:
                    continue
                
                # Áp dụng augmentation ngẫu nhiên
                aug_img, aug_type = self._apply_augmentation(img)
                
                # Tạo tên file mới
                base_name = Path(row['fname']).stem
                ext = Path(row['fname']).suffix
                new_fname = f"{base_name}_aug_{aug_count}_{aug_type}{ext}"
                new_path = os.path.join(aug_dir, new_fname)
                
                # Lưu ảnh augmented
                cv2.imwrite(new_path, aug_img)
                
                # Tạo row mới với thông tin cập nhật
                new_row = row.copy()
                new_row['fname'] = new_fname
                new_row['augmented'] = True
                new_row['aug_source'] = row['fname']
                new_row['aug_path'] = new_path
                
                # Cập nhật bbox nếu có flip
                if 'flip' in aug_type and pd.notna(row.get('bbox')) and row['bbox'] != 'none':
                    new_row['bbox'] = self._flip_bbox(row['bbox'], img.shape[1])
                
                new_rows.append(new_row)
                aug_count += 1
        
        # Thêm ảnh augmented vào dataframe
        if new_rows:
            aug_df = pd.DataFrame(new_rows)
            self.df = pd.concat([self.df, aug_df], ignore_index=True)
            
        print(f"   ✅ Đã tạo {aug_count} ảnh augmented cho lớp '{target_class}'")
        print(f"   📊 Tổng số ảnh lớp '{target_class}' sau augment: {len(self.df[self.df['class_name'] == target_class])}")
    
    def _apply_augmentation(self, img: np.ndarray) -> tuple:
        """
        Áp dụng augmentation ngẫu nhiên cho ảnh.
        
        Các kỹ thuật được chọn phù hợp với ảnh X-quang y tế:
        - Rotation nhẹ (±15°): Mô phỏng góc chụp khác nhau
        - Brightness: Mô phỏng điều kiện chụp khác nhau
        - Horizontal flip: Vẫn giữ đặc điểm y học
        - Contrast: Tăng/giảm độ tương phản
        
        Returns:
            tuple: (ảnh augmented, loại augmentation)
        """
        aug_type = random.choice(['rotate', 'brightness', 'flip', 'contrast', 'combined'])
        
        if aug_type == 'rotate':
            # Xoay ±15 độ
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            
        elif aug_type == 'brightness':
            # Thay đổi độ sáng
            factor = random.uniform(0.7, 1.3)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            
        elif aug_type == 'flip':
            # Lật ngang
            img = cv2.flip(img, 1)
            
        elif aug_type == 'contrast':
            # Thay đổi contrast
            factor = random.uniform(0.8, 1.2)
            mean = np.mean(img)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=(1 - factor) * mean)
            
        elif aug_type == 'combined':
            # Kết hợp nhiều augmentation
            # Flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                aug_type = 'combined_flip'
            # Brightness
            factor = random.uniform(0.8, 1.2)
            img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            # Rotation nhẹ
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return img, aug_type
    
    def _flip_bbox(self, bbox_str: str, img_width: int) -> str:
        """Lật bbox theo chiều ngang khi flip ảnh"""
        import ast
        try:
            bbox = ast.literal_eval(bbox_str)
            # Tính lại xmin sau khi flip
            new_xmin = img_width - bbox['xmin'] - bbox['width']
            bbox['xmin'] = new_xmin
            return str(bbox)
        except:
            return bbox_str
    
    # =========================================================================
    # TẠO DATASET CHO CLASSIFICATION (dataset_cls)
    # =========================================================================
    
    def create_classification_dataset(self):
        """
        Tạo dataset cho YOLOv8-cls.
        
        Cấu trúc output:
        dataset_cls/
        ├── train/
        │   ├── healthy/
        │   ├── sick_but_no_tb/
        │   ├── active_tb/
        │   └── latent_tb/
        └── val/
            ├── healthy/
            ├── sick_but_no_tb/
            ├── active_tb/
            └── latent_tb/
        
        Logic:
        - Sử dụng TẤT CẢ ảnh từ 4 lớp
        - Chia 80% train, 20% val (stratified)
        """
        print("\n" + "="*60)
        print("📁 TẠO DATASET CLASSIFICATION (4 LỚP)")
        print("="*60)
        
        # Tạo thư mục
        for split in ['train', 'val']:
            for cls in self.cls_classes:
                os.makedirs(os.path.join(self.cls_dir, split, cls), exist_ok=True)
        
        # Chia train/val theo stratified sampling
        train_df, val_df = train_test_split(
            self.df, 
            test_size=self.val_ratio,
            stratify=self.df['class_name'],
            random_state=42
        )
        
        print(f"\n📊 Phân chia dữ liệu:")
        print(f"   Train: {len(train_df)} ảnh")
        print(f"   Val: {len(val_df)} ảnh")
        
        # Copy ảnh vào thư mục tương ứng
        stats = {'train': {cls: 0 for cls in self.cls_classes}, 
                 'val': {cls: 0 for cls in self.cls_classes}}
        
        for split, split_df in [('train', train_df), ('val', val_df)]:
            print(f"\n📦 Đang xử lý {split}...")
            for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"   {split}"):
                cls_name = row['class_name']
                
                # Xác định đường dẫn nguồn
                if row.get('augmented', False) and pd.notna(row.get('aug_path')):
                    src_path = str(row['aug_path'])
                else:
                    src_path = self.find_image_path(str(row['fname']))
                
                if src_path and os.path.exists(src_path):
                    dst_path = os.path.join(self.cls_dir, split, cls_name, str(row['fname']))
                    shutil.copy2(src_path, dst_path)
                    stats[split][cls_name] += 1
        
        # In thống kê
        print(f"\n✅ Dataset Classification đã sẵn sàng!")
        print(f"📁 Đường dẫn: {self.cls_dir}")
        print(f"\n📊 Thống kê chi tiết:")
        for split in ['train', 'val']:
            print(f"\n   {split.upper()}:")
            for cls in self.cls_classes:
                print(f"      {cls}: {stats[split][cls]}")
        
        # Tạo file yaml config
        self._create_cls_yaml()
        
        return self.cls_dir
    
    def _create_cls_yaml(self):
        """Tạo file YAML config cho YOLOv8-cls"""
        yaml_content = {
            'path': os.path.abspath(self.cls_dir),
            'train': 'train',
            'val': 'val',
            'nc': 4,
            'names': self.cls_classes
        }
        
        yaml_path = os.path.join(self.cls_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"\n📝 File config: {yaml_path}")
    
    # =========================================================================
    # TẠO DATASET CHO DETECTION (dataset_det)
    # =========================================================================
    
    def create_detection_dataset(self, bg_samples_per_class: int = 500):
        """
        Tạo dataset cho YOLOv8-det.
        
        Cấu trúc output:
        dataset_det/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── dataset.yaml
        
        Logic quan trọng:
        
        1. POSITIVE SAMPLES (có bounding box):
           - active_tb: class_id = 0
           - latent_tb: class_id = 1
           - File label: chứa bbox theo format YOLO
        
        2. BACKGROUND IMAGES (không có object):
           - healthy và sick_but_no_tb
           - Lấy mẫu một số lượng nhất định (mặc định 500 mỗi lớp)
           - File label: TẠO FILE RỖNG (.txt trống)
           
           ⚠️ TẠI SAO CẦN BACKGROUND IMAGES?
           - YOLO cần học phân biệt "có object" vs "không có object"
           - Nếu chỉ train với ảnh có TB, model sẽ có xu hướng 
             phát hiện TB ở mọi nơi (False Positive cao)
           - Background images dạy model biết rằng:
             "Đây là ảnh KHÔNG có tổn thương TB"
           - File .txt rỗng = không có object trong ảnh này
        
        Args:
            bg_samples_per_class: Số ảnh background lấy từ mỗi lớp (healthy, sick_but_no_tb)
        """
        print("\n" + "="*60)
        print("📁 TẠO DATASET DETECTION (2 LỚP + BACKGROUND)")
        print("="*60)
        
        # Tạo thư mục
        for folder in ['images', 'labels']:
            for split in ['train', 'val']:
                os.makedirs(os.path.join(self.det_dir, folder, split), exist_ok=True)
        
        # =====================================================================
        # BƯỚC 1: XỬ LÝ POSITIVE SAMPLES (active_tb, latent_tb)
        # =====================================================================
        print("\n📌 Bước 1: Xử lý Positive Samples (có bounding box)...")
        
        # Lọc ảnh có bbox (active_tb và latent_tb)
        positive_df = self.df[self.df['class_name'].isin(['active_tb', 'latent_tb'])].copy()
        
        # Chia train/val
        pos_train, pos_val = train_test_split(
            positive_df,
            test_size=self.val_ratio,
            stratify=positive_df['class_name'],
            random_state=42
        )
        
        stats = {
            'train': {'active_tb': 0, 'latent_tb': 0, 'background': 0},
            'val': {'active_tb': 0, 'latent_tb': 0, 'background': 0}
        }
        
        # Xử lý positive samples
        for split, split_df in [('train', pos_train), ('val', pos_val)]:
            print(f"\n   📦 Đang xử lý {split} (positive)...")
            for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"      {split}"):
                success = self._process_positive_sample(row, split)
                if success:
                    stats[split][row['class_name']] += 1
        
        # =====================================================================
        # BƯỚC 2: XỬ LÝ BACKGROUND IMAGES (healthy, sick_but_no_tb)
        # =====================================================================
        print(f"\n📌 Bước 2: Xử lý Background Images (không có object)...")
        print(f"   ℹ️  Lấy {bg_samples_per_class} ảnh từ mỗi lớp: healthy, sick_but_no_tb")
        print(f"""
   ⚠️  TẠI SAO CẦN BACKGROUND IMAGES?
   ────────────────────────────────────────
   • YOLO cần học phân biệt "có object" vs "không có object"
   • Nếu chỉ train với ảnh có TB → False Positive cao!
   • Background images dạy model: "Ảnh này KHÔNG có tổn thương TB"
   • File .txt rỗng = không có object nào trong ảnh này
   ────────────────────────────────────────
""")
        
        # Lọc ảnh không có TB
        background_df = self.df[self.df['class_name'].isin(['healthy', 'sick_but_no_tb'])].copy()
        
        # Lấy mẫu từ mỗi lớp
        bg_samples = []
        for cls in ['healthy', 'sick_but_no_tb']:
            cls_df = background_df[background_df['class_name'] == cls]
            sample_size = min(bg_samples_per_class, len(cls_df))
            sampled = cls_df.sample(n=sample_size, random_state=42)
            bg_samples.append(sampled)
            print(f"   Đã lấy {sample_size} ảnh từ lớp '{cls}'")
        
        bg_df = pd.concat(bg_samples, ignore_index=True)
        
        # Chia train/val cho background
        bg_train, bg_val = train_test_split(
            bg_df,
            test_size=self.val_ratio,
            random_state=42
        )
        
        # Xử lý background samples
        for split, split_df in [('train', bg_train), ('val', bg_val)]:
            print(f"\n   📦 Đang xử lý {split} (background)...")
            for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"      {split}"):
                success = self._process_background_sample(row, split)
                if success:
                    stats[split]['background'] += 1
        
        # In thống kê
        print(f"\n✅ Dataset Detection đã sẵn sàng!")
        print(f"📁 Đường dẫn: {self.det_dir}")
        print(f"\n📊 Thống kê chi tiết:")
        for split in ['train', 'val']:
            print(f"\n   {split.upper()}:")
            print(f"      active_tb (class 0): {stats[split]['active_tb']}")
            print(f"      latent_tb (class 1): {stats[split]['latent_tb']}")
            print(f"      background (no object): {stats[split]['background']}")
            total = sum(stats[split].values())
            print(f"      ─────────────────────")
            print(f"      Tổng: {total}")
        
        # Tạo file yaml config
        self._create_det_yaml()
        
        return self.det_dir
    
    def _process_positive_sample(self, row, split: str) -> bool:
        """
        Xử lý một ảnh positive (có bounding box).
        
        Args:
            row: Dòng dữ liệu từ DataFrame
            split: 'train' hoặc 'val'
        
        Returns:
            bool: True nếu xử lý thành công
        """
        import ast
        
        # Xác định đường dẫn nguồn
        if row.get('augmented', False) and pd.notna(row.get('aug_path')):
            src_path = str(row['aug_path'])
        else:
            src_path = self.find_image_path(str(row['fname']))
        
        if not src_path or not os.path.exists(src_path):
            return False
        
        # Copy ảnh
        dst_img = os.path.join(self.det_dir, 'images', split, str(row['fname']))
        shutil.copy2(src_path, dst_img)
        
        # Tạo file label
        label_fname = Path(str(row['fname'])).stem + '.txt'
        label_path = os.path.join(self.det_dir, 'labels', split, label_fname)
        
        # Parse bbox
        try:
            bbox_str = row.get('bbox', 'none')
            if pd.isna(bbox_str) or bbox_str == 'none':
                # Tạo file rỗng nếu không có bbox
                open(label_path, 'w').close()
                return True
            
            bbox = ast.literal_eval(bbox_str)
            img_w = float(row['image_width'])
            img_h = float(row['image_height'])
            
            # Chuyển sang format YOLO: class_id x_center y_center width height (normalized)
            x_center = (bbox['xmin'] + bbox['width'] / 2) / img_w
            y_center = (bbox['ymin'] + bbox['height'] / 2) / img_h
            width = bbox['width'] / img_w
            height = bbox['height'] / img_h
            
            # Clamp values
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Class ID: 0 = active_tb, 1 = latent_tb
            class_id = 0 if row['class_name'] == 'active_tb' else 1
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            return True
            
        except Exception as e:
            # Tạo file rỗng nếu lỗi
            open(label_path, 'w').close()
            return True
    
    def _process_background_sample(self, row, split: str) -> bool:
        """
        Xử lý một ảnh background (không có object).
        
        QUAN TRỌNG: Tạo file .txt RỖNG để YOLO biết rằng
        ảnh này không chứa object nào.
        
        Args:
            row: Dòng dữ liệu từ DataFrame
            split: 'train' hoặc 'val'
        
        Returns:
            bool: True nếu xử lý thành công
        """
        src_path = self.find_image_path(row['fname'])
        
        if not src_path or not os.path.exists(src_path):
            return False
        
        # Copy ảnh
        dst_img = os.path.join(self.det_dir, 'images', split, row['fname'])
        shutil.copy2(src_path, dst_img)
        
        # TẠO FILE LABEL RỖNG
        # ⚠️ Đây là bước quan trọng! File .txt rỗng cho YOLO biết
        # rằng ảnh này không chứa bất kỳ object nào.
        label_fname = Path(str(row['fname'])).stem + '.txt'
        label_path = os.path.join(self.det_dir, 'labels', split, label_fname)
        
        # Tạo file rỗng
        open(label_path, 'w').close()
        
        return True
    
    def _create_det_yaml(self):
        """Tạo file YAML config cho YOLOv8-det"""
        yaml_content = {
            'path': os.path.abspath(self.det_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 2,
            'names': self.det_classes
        }
        
        yaml_path = os.path.join(self.det_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"\n📝 File config: {yaml_path}")
    
    # =========================================================================
    # CHẠY TOÀN BỘ PIPELINE
    # =========================================================================
    
    def run(self, augment_latent_tb: bool = True, latent_target: int = 800,
            bg_samples_per_class: int = 500):
        """
        Chạy toàn bộ pipeline chuẩn bị dữ liệu.
        
        Args:
            augment_latent_tb: Có augment lớp latent_tb không
            latent_target: Số lượng ảnh mục tiêu cho latent_tb sau augment
            bg_samples_per_class: Số ảnh background lấy từ mỗi lớp
        """
        print("\n" + "="*60)
        print("🚀 BẮT ĐẦU CHUẨN BỊ DỮ LIỆU CHO YOLOv8")
        print("="*60)
        
        # 1. Load data
        self.load_data()
        
        # 2. Offline augmentation cho lớp thiểu số
        if augment_latent_tb:
            self.offline_augmentation('latent_tb', target_count=latent_target)
        
        # 3. Tạo dataset classification
        self.create_classification_dataset()
        
        # 4. Tạo dataset detection
        self.create_detection_dataset(bg_samples_per_class=bg_samples_per_class)
        
        # 5. In hướng dẫn training
        self._print_training_instructions()
        
        print("\n" + "="*60)
        print("✅ HOÀN TẤT CHUẨN BỊ DỮ LIỆU!")
        print("="*60)
    
    def _print_training_instructions(self):
        """In hướng dẫn training"""
        print("\n" + "="*60)
        print("📖 HƯỚNG DẪN TRAINING")
        print("="*60)
        
        print("""
╔══════════════════════════════════════════════════════════════╗
║  STAGE 1: CLASSIFICATION (4 LỚP)                             ║
╠══════════════════════════════════════════════════════════════╣
║  Command:                                                     ║
║  yolo classify train \\                                       ║
║      data={cls_dir} \\                                        ║
║      model=yolov8n-cls.pt \\                                  ║
║      epochs=100 \\                                            ║
║      imgsz=224 \\                                             ║
║      batch=32 \\                                              ║
║      project=tb_classification                                ║
║                                                               ║
║  Hoặc dùng Python:                                            ║
║  from ultralytics import YOLO                                 ║
║  model = YOLO('yolov8n-cls.pt')                              ║
║  model.train(data='{cls_dir}', epochs=100, imgsz=224)        ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  STAGE 2: DETECTION (2 LỚP + BACKGROUND)                     ║
╠══════════════════════════════════════════════════════════════╣
║  Command:                                                     ║
║  yolo detect train \\                                         ║
║      data={det_yaml} \\                                       ║
║      model=yolov8n.pt \\                                      ║
║      epochs=100 \\                                            ║
║      imgsz=640 \\                                             ║
║      batch=16 \\                                              ║
║      project=tb_detection                                     ║
║                                                               ║
║  Hoặc dùng Python:                                            ║
║  from ultralytics import YOLO                                 ║
║  model = YOLO('yolov8n.pt')                                  ║
║  model.train(data='{det_yaml}', epochs=100, imgsz=640)       ║
╚══════════════════════════════════════════════════════════════╝
""".format(
            cls_dir=self.cls_dir,
            det_yaml=os.path.join(self.det_dir, 'dataset.yaml')
        ))


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Hàm main để chạy từ command line"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Chuẩn bị dữ liệu cho YOLOv8 Classification và Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python prepare_data.py --csv data.csv --images ./images --output ./datasets
  python prepare_data.py --csv data.csv --images ./images --output ./datasets --no-augment
  python prepare_data.py --csv data.csv --images ./images --output ./datasets --bg-samples 800
        """
    )
    
    parser.add_argument('--csv', type=str, required=True,
                        help='Đường dẫn đến file CSV chứa metadata')
    parser.add_argument('--images', type=str, required=True,
                        help='Thư mục chứa ảnh gốc')
    parser.add_argument('--output', type=str, default='./datasets',
                        help='Thư mục output (mặc định: ./datasets)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Tỷ lệ validation (mặc định: 0.2)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Không thực hiện offline augmentation')
    parser.add_argument('--latent-target', type=int, default=800,
                        help='Số lượng ảnh mục tiêu cho latent_tb sau augment (mặc định: 800)')
    parser.add_argument('--bg-samples', type=int, default=500,
                        help='Số ảnh background lấy từ mỗi lớp (mặc định: 500)')
    
    args = parser.parse_args()
    
    # Chạy pipeline
    preparer = DataPreparer(
        csv_path=args.csv,
        images_dir=args.images,
        output_dir=args.output,
        val_ratio=args.val_ratio
    )
    
    preparer.run(
        augment_latent_tb=not args.no_augment,
        latent_target=args.latent_target,
        bg_samples_per_class=args.bg_samples
    )


if __name__ == '__main__':
    main()
