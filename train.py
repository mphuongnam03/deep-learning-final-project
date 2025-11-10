"""
Huấn luyện YOLOv8 model cho TB Detection
"""

from ultralytics import YOLO
import torch
import yaml
from datetime import datetime

class TBModelTrainer:
    def __init__(self, data_yaml, model_size='n', project='tb_detection'):
        """
        Args:
            data_yaml: Đường dẫn đến dataset.yaml
            model_size: 'n', 's', 'm', 'l', 'x' (nano đến xlarge)
            project: Tên project để lưu results
        """
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.project = project
        
        # Load model
        self.model = YOLO(f'yolov8{model_size}.pt')
        print(f"✅ Đã load YOLOv8{model_size} model")
    
    def train(self, epochs=100, batch_size=16, img_size=512, device=0):
        """Huấn luyện model"""
        
        print("\n" + "="*50)
        print("🚀 BẮT ĐẦU HUẤN LUYỆN")
        print("="*50)
        
        # Training parameters
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            
            # Project settings
            project=self.project,
            name=f'yolov8{self.model_size}_tb_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=True,
            
            # Training settings
            patience=20,  # Early stopping
            save=True,
            save_period=10,  # Save mỗi 10 epochs
            
            # Optimization
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            
            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Validation
            val=True,
            plots=True,
            
            # Verbose
            verbose=True,
        )
        
        print("\n✅ HUẤN LUYỆN HOÀN TẤT!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        return results
    
    def resume_training(self, checkpoint_path):
        """Tiếp tục huấn luyện từ checkpoint"""
        self.model = YOLO(checkpoint_path)
        return self.train()