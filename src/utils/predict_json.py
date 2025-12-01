"""
Script để chạy inference YOLO và trả về kết quả dưới dạng JSON
Bao gồm bounding boxes, confidence scores, và ảnh đã được đánh dấu
"""

import json
import base64
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


class TBDetectionAPI:
    def __init__(self, model_path, model_version="yolov8n-v1.0"):
        """
        Args:
            model_path: Đường dẫn đến file model (.pt)
            model_version: Phiên bản model
        """
        self.model = YOLO(model_path)
        self.model_version = model_version
        
    def predict_image(self, image_path, conf_threshold=0.25, save_annotated=True):
        """
        Dự đoán trên một ảnh và trả về kết quả dưới dạng JSON
        
        Args:
            image_path: Đường dẫn đến ảnh
            conf_threshold: Ngưỡng confidence
            save_annotated: Có lưu ảnh đã đánh dấu không
            
        Returns:
            dict: Kết quả dưới dạng JSON
        """
        # Load ảnh
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        # Predict
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )
        
        # Parse kết quả
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 2),
                    "class_id": class_id,
                    "class_name": class_name
                })
        
        # Vẽ bounding boxes lên ảnh
        annotated_img = self._draw_boxes(img.copy(), detections)
        
        # Convert ảnh sang base64
        annotated_img_base64 = self._image_to_base64(annotated_img)
        
        # Tạo JSON response
        response = {
            "model_version": self.model_version,
            "image_name": Path(image_path).name,
            "image_size": {
                "width": img.shape[1],
                "height": img.shape[0]
            },
            "detections": detections,
            "annotated_image": annotated_img_base64
        }
        
        # Lưu ảnh đã đánh dấu nếu cần
        if save_annotated:
            output_path = f"output_annotated_{Path(image_path).name}"
            cv2.imwrite(output_path, annotated_img)
            response["annotated_image_path"] = output_path
        
        return response
    
    def _draw_boxes(self, img, detections):
        """Vẽ bounding boxes lên ảnh"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            class_name = det["class_name"]
            
            # Chọn màu theo class (4 classes)
            if class_name == "healthy":
                color = (0, 255, 0)  # Green
            elif class_name == "sick_but_no_tb":
                color = (0, 165, 255)  # Orange
            elif class_name == "active_tb":
                color = (0, 0, 255)  # Red
            else:  # latent_tb
                color = (255, 0, 255)  # Magenta
            
            # Vẽ bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label
            label = f"{class_name} {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(img, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def _image_to_base64(self, img):
        """Convert OpenCV image sang base64 string"""
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def predict_batch(self, images_dir, output_json="results.json", conf_threshold=0.25):
        """
        Dự đoán trên nhiều ảnh và lưu kết quả vào file JSON
        
        Args:
            images_dir: Thư mục chứa ảnh
            output_json: Đường dẫn file JSON output
            conf_threshold: Ngưỡng confidence
        """
        results = []
        image_files = list(Path(images_dir).glob('*.png')) + \
                     list(Path(images_dir).glob('*.jpg')) + \
                     list(Path(images_dir).glob('*.jpeg'))
        
        for img_path in image_files:
            print(f"Processing {img_path.name}...")
            try:
                result = self.predict_image(str(img_path), conf_threshold)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        # Lưu vào file JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Đã lưu kết quả vào: {output_json}")
        return results


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Khởi tạo API
    detector = TBDetectionAPI(
        model_path="tb_detection/yolov8n_tb_20251011_190341/weights/best.pt",
        model_version="yolov8n-tb-v1.0"
    )
    
    # Dự đoán trên một ảnh
    result = detector.predict_image(
        image_path="test_images/sample.png",
        conf_threshold=0.25
    )
    
    # In kết quả JSON
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Lưu vào file
    with open("prediction_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Kết quả đã được lưu vào prediction_result.json")
    
    # Hoặc dự đoán trên nhiều ảnh
    # detector.predict_batch("test_images", "batch_results.json")
