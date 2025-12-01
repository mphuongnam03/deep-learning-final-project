"""
Tầng Hạ Tầng - Adapter YOLOv8

Triển khai cụ thể các cổng interface sử dụng Ultralytics YOLOv8.
Các adapter này bao bọc thư viện YOLO và tuân thủ các interface đã định nghĩa.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from ultralytics import YOLO

from src.domain.entities import (
    ClassificationResult,
    DetectionResult,
    BoundingBox,
    DiagnosisClass
)
from src.interfaces.ports import ImageClassifierPort, ObjectDetectorPort


class YOLOv8Classifier(ImageClassifierPort):
    """
    Adapter phân loại YOLOv8.
    Bao bọc model YOLOv8-cls để phân loại ảnh X-quang ngực 4 lớp.
    """
    
    # Ánh xạ từ chỉ số lớp sang enum DiagnosisClass
    CLASS_INDEX_MAP = {
        0: DiagnosisClass.HEALTHY,
        1: DiagnosisClass.SICK_BUT_NO_TB,
        2: DiagnosisClass.ACTIVE_TB,
        3: DiagnosisClass.LATENT_TB
    }
    
    def __init__(self, model_path: Optional[Path] = None):
        self._model: Optional[YOLO] = None
        self._class_names: List[str] = [
            "healthy", "sick_but_no_tb", "active_tb", "latent_tb"
        ]
        
        if model_path:
            self.load_model(model_path)
    
    @property
    def class_names(self) -> List[str]:
        return self._class_names
    
    @property
    def num_classes(self) -> int:
        return len(self._class_names)
    
    def load_model(self, model_path: Path) -> None:
        """Tải trọng số model YOLOv8-cls."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
        
        try:
            self._model = YOLO(str(model_path))
            # Kiểm tra đây là model phân loại bằng cách kiểm tra task
            if hasattr(self._model, 'task') and self._model.task != 'classify':
                print(f"⚠️ Cảnh báo: Task của model là '{self._model.task}', mong đợi 'classify'")
            print(f"✅ Đã tải model phân loại: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Không thể tải model phân loại: {e}")
    
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def classify(
        self,
        image: np.ndarray,
        return_probabilities: bool = False
    ) -> ClassificationResult:
        """
        Phân loại ảnh X-quang ngực vào 1 trong 4 lớp chẩn đoán.
        """
        if not self.is_loaded():
            raise RuntimeError("Model chưa được tải. Vui lòng gọi load_model() trước.")
        
        # Chạy suy luận
        results = self._model.predict(source=image, verbose=False)
        
        if len(results) == 0:
            raise RuntimeError("Phân loại không trả về kết quả")
        
        result = results[0]
        
        # Trích xuất dự đoán - YOLOv8-cls lưu kết quả trong thuộc tính probs
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        
        # Ánh xạ sang enum DiagnosisClass
        predicted_class = self.CLASS_INDEX_MAP.get(top1_idx, DiagnosisClass.HEALTHY)
        
        # Xây dựng dict xác suất nếu được yêu cầu
        all_probs = None
        if return_probabilities:
            probs_data = probs.data.cpu().numpy()
            all_probs = {
                self._class_names[i]: round(float(probs_data[i]), 4)
                for i in range(len(self._class_names))
            }
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=top1_conf,
            all_probabilities=all_probs
        )


class YOLOv8Detector(ObjectDetectorPort):
    """
    Adapter phát hiện YOLOv8.
    Bao bọc model phát hiện YOLOv8 để định vị tổn thương TB.
    Chỉ phát hiện 2 lớp: active_tb và latent_tb.
    """
    
    # Ánh xạ lớp của model phát hiện
    CLASS_NAMES = ["active_tb", "latent_tb"]
    
    def __init__(self, model_path: Optional[Path] = None):
        self._model: Optional[YOLO] = None
        
        if model_path:
            self.load_model(model_path)
    
    @property
    def class_names(self) -> List[str]:
        return self.CLASS_NAMES
    
    def load_model(self, model_path: Path) -> None:
        """Tải trọng số model phát hiện YOLOv8."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
        
        try:
            self._model = YOLO(str(model_path))
            print(f"✅ Đã tải model phát hiện: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Không thể tải model phát hiện: {e}")
    
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> DetectionResult:
        """
        Phát hiện vùng tổn thương TB trong ảnh.
        """
        if not self.is_loaded():
            raise RuntimeError("Model chưa được tải. Vui lòng gọi load_model() trước.")
        
        # Chạy suy luận với các ngưỡng đã chỉ định
        results = self._model.predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        bounding_boxes: List[BoundingBox] = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Trích xuất tọa độ
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Ánh xạ class_id sang tên lớp
                # Model phát hiện sử dụng: 0 = active_tb, 1 = latent_tb
                class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else "unknown"
                
                bounding_boxes.append(BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_name=class_name
                ))
        
        return DetectionResult(
            bounding_boxes=bounding_boxes,
            num_lesions=len(bounding_boxes)
        )
