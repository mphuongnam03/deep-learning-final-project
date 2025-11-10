"""
Đánh giá và test model
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class TBModelEvaluator:
    def __init__(self, model_path):
        """
        Args:
            model_path: Đường dẫn đến best.pt hoặc last.pt
        """
        self.model = YOLO(model_path)
        print(f"✅ Đã load model: {model_path}")
    
    def evaluate_on_validation(self, data_yaml):
        """Đánh giá trên validation set"""
        print("\n📊 ĐÁNH GIÁ TRÊN VALIDATION SET")
        print("="*50)
        
        metrics = self.model.val(data=data_yaml, split='val')
        
        print(f"\n🎯 KẾT QUẢ:")
        print(f"   mAP50:     {metrics.box.map50:.4f}")
        print(f"   mAP50-95:  {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall:    {metrics.box.mr:.4f}")
        
        return metrics
    
    def predict_single_image(self, image_path, conf=0.25, save=True):
        """Dự đoán trên 1 ảnh"""
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=save,
            save_txt=True,
            save_conf=True,
        )
        
        return results[0]
    
    def predict_batch(self, images_dir, conf=0.25, save_dir='runs/predict'):
        """Dự đoán trên nhiều ảnh"""
        results = self.model.predict(
            source=images_dir,
            conf=conf,
            save=True,
            project=save_dir,
        )
        
        return results