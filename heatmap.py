"""
Tạo heatmap visualization cho TB detection
"""

import cv2
import numpy as np
from ultralytics import YOLO

class TBHeatmapGenerator:
    def __init__(self, model_path):
        """
        Args:
            model_path: Đường dẫn đến model.pt
        """
        self.model = YOLO(model_path)
    
    def generate_heatmap(self, image_path, output_path=None, conf=0.25):
        """
        Tạo heatmap từ bounding boxes predictions
        
        Args:
            image_path: Đường dẫn ảnh input
            output_path: Đường dẫn lưu output
            conf: Confidence threshold
        """
        # Load ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        h, w = img.shape[:2]
        
        # Predict
        results = self.model.predict(image_path, conf=conf, verbose=False)
        
        # Tạo heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Chỉ tạo heatmap cho class TB (class_id=2)
                if cls == 2:
                    # Tạo gradient từ tâm bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    for i in range(max(0, y1), min(h, y2)):
                        for j in range(max(0, x1), min(w, x2)):
                            # Khoảng cách từ pixel đến tâm
                            dist = np.sqrt((j - center_x)**2 + (i - center_y)**2)
                            max_dist = np.sqrt(((x2-x1)/2)**2 + ((y2-y1)/2)**2)
                            
                            if max_dist > 0:
                                # Intensity giảm dần từ tâm
                                intensity = (1 - dist/max_dist) * confidence
                                heatmap[i, j] = max(heatmap[i, j], intensity)
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Apply colormap (JET: xanh lạnh -> đỏ nóng)
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Overlay lên ảnh gốc
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        
        # Vẽ bounding boxes và labels
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Color theo class
                if cls == 0:  # healthy
                    color = (0, 255, 0)  # Green
                elif cls == 1:  # sick_but_no_tb
                    color = (0, 165, 255)  # Orange
                else:  # tb
                    color = (0, 0, 255)  # Red
                
                # Vẽ bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{self.model.names[cls]} {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(overlay, (x1, y1-label_h-10), 
                            (x1+label_w, y1), color, -1)
                cv2.putText(overlay, label, (x1, y1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save nếu có output_path
        if output_path:
            cv2.imwrite(output_path, overlay)
            print(f"✅ Đã lưu heatmap: {output_path}")
        
        return overlay, heatmap
    
    def generate_batch_heatmaps(self, images_dir, output_dir, conf=0.25):
        """Tạo heatmap cho nhiều ảnh"""
        import os
        from pathlib import Path
        
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = list(Path(images_dir).glob('*.png')) + \
                     list(Path(images_dir).glob('*.jpg'))
        
        for img_path in image_files:
            output_path = os.path.join(output_dir, f"heatmap_{img_path.name}")
            self.generate_heatmap(str(img_path), output_path, conf)
        
        print(f"✅ Đã tạo {len(image_files)} heatmaps trong {output_dir}")
