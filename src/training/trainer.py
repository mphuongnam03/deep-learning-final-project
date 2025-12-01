"""
Huấn luyện model YOLOv8 cho Phát hiện Bệnh Lao Phổi
"""

from ultralytics import YOLO
import torch
import yaml
from datetime import datetime

class TBModelTrainer:
    def __init__(self, data_yaml, model_size='n', project='tb_detection'):
        """
        Khởi tạo bộ huấn luyện.
        
        Args:
            data_yaml: Đường dẫn đến file dataset.yaml
            model_size: 'n', 's', 'm', 'l', 'x' (nano đến xlarge)
            project: Tên project để lưu kết quả
        """
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.project = project
        
        # Tải model
        self.model = YOLO(f'yolov8{model_size}.pt')
        print(f"✅ Đã tải model YOLOv8{model_size}")
    
    def train(self, epochs=100, batch_size=16, img_size=512, device=0):
        """Huấn luyện model"""
        
        print("\n" + "="*50)
        print("🚀 BẮT ĐẦU HUẤN LUYỆN")
        print("="*50)
        
        # Các tham số huấn luyện
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            
            # Cài đặt project
            project=self.project,
            name=f'yolov8{self.model_size}_tb_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=True,
            
            # Cài đặt huấn luyện
            patience=20,  # Dừng sớm nếu không cải thiện
            save=True,
            save_period=10,  # Lưu mỗi 10 epochs
            
            # Tối ưu hóa
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Tăng cường dữ liệu - Để xử lý mất cân bằng
            hsv_h=0.02,      
            hsv_s=0.8,      
            hsv_v=0.5, 
            degrees=0.0,
            translate=0.15,  # Tăng từ 0.1
            scale=0.6,       # Tăng từ 0.5
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15, 
            copy_paste=0.1,
            
            # Trọng số loss - Tăng weight cho phân loại để xử lý mất cân bằng
            box=7.5,
            cls=1.0,
            dfl=1.5,
            
            # Xác thực
            val=True,
            plots=True,
            
            # Hiển thị chi tiết
            verbose=True,
        )
        
        print("\n✅ HUẤN LUYỆN HOÀN TẤT!")
        print(f"📁 Kết quả lưu tại: {results.save_dir}")
        
        return results
    
    def resume_training(self, checkpoint_path):
        """Tiếp tục huấn luyện từ checkpoint"""
        self.model = YOLO(checkpoint_path)
        return self.train()