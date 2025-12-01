"""
Tầng Hạ Tầng - Tiện ích Xử Lý Ảnh

Triển khai các tiện ích tải ảnh, chú thích và chuyển đổi.
"""

import base64
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

from src.domain.entities import BoundingBox, DiagnosisClass
from src.interfaces.ports import ImageAnnotatorPort, ImageLoaderPort


class OpenCVImageLoader(ImageLoaderPort):
    """
    Triển khai tải ảnh sử dụng OpenCV.
    Hỗ trợ các định dạng ảnh phổ biến: PNG, JPG, JPEG, BMP, v.v.
    """
    
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    def load(self, source: str) -> Tuple[np.ndarray, int, int]:
        """Tải ảnh từ đường dẫn file."""
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {source}")
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Định dạng ảnh không được hỗ trợ: {path.suffix}")
        
        image = cv2.imread(str(path))
        
        if image is None:
            raise ValueError(f"Không thể tải ảnh: {source}")
        
        height, width = image.shape[:2]
        return image, width, height
    
    def validate(self, source: str) -> bool:
        """Kiểm tra xem nguồn có phải là file ảnh hợp lệ không."""
        try:
            path = Path(source)
            return (
                path.exists() and 
                path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            )
        except Exception:
            return False


class TBAnnotator(ImageAnnotatorPort):
    """
    Trình chú thích ảnh cho kết quả chẩn đoán TB.
    Vẽ nhãn phân loại và bounding box với kiểu dáng mã màu.
    """
    
    # Bảng màu cho mỗi lớp chẩn đoán (định dạng BGR cho OpenCV)
    CLASS_COLORS = {
        DiagnosisClass.HEALTHY: (0, 255, 0),          # Xanh lá - Khỏe mạnh
        DiagnosisClass.SICK_BUT_NO_TB: (0, 255, 255), # Vàng - Bệnh nhưng không phải TB
        DiagnosisClass.ACTIVE_TB: (0, 0, 255),        # Đỏ - Lao hoạt động
        DiagnosisClass.LATENT_TB: (0, 165, 255)       # Cam - Lao tiềm ẩn
    }
    
    # Màu sắc của box phát hiện (các tông khác nhau cho active và latent)
    BOX_COLORS = {
        "active_tb": (0, 0, 255),    # Đỏ
        "latent_tb": (0, 165, 255)   # Cam
    }
    
    def annotate(
        self,
        image: np.ndarray,
        diagnosis_class: DiagnosisClass,
        bounding_boxes: List[BoundingBox],
        classification_confidence: float
    ) -> np.ndarray:
        """
        Vẽ các chú thích chẩn đoán lên ảnh.
        
        Bao gồm:
        - Banner trên cùng với kết quả phân loại
        - Bounding box cho các tổn thương được phát hiện (nếu TB dương tính)
        - Điểm tin cậy cho mỗi phát hiện
        """
        annotated = image.copy()
        height, width = annotated.shape[:2]
        
        # Lấy màu cho lớp chẩn đoán này
        color = self.CLASS_COLORS.get(diagnosis_class, (128, 128, 128))
        
        # Vẽ banner phân loại ở trên cùng
        banner_height = 60
        cv2.rectangle(annotated, (0, 0), (width, banner_height), color, -1)
        
        # Thêm text phân loại
        class_text = f"{diagnosis_class.value.upper()}"
        conf_text = f"Confidence: {classification_confidence:.1%}"
        
        cv2.putText(
            annotated, class_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            annotated, conf_text, (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # Vẽ số lượng tổn thương nếu TB dương tính
        if bounding_boxes:
            lesion_text = f"Lesions: {len(bounding_boxes)}"
            text_size = cv2.getTextSize(lesion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(
                annotated, lesion_text, (width - text_size[0] - 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        # Vẽ bounding box cho mỗi phát hiện
        for box in bounding_boxes:
            box_color = self.BOX_COLORS.get(box.class_name, (0, 255, 255))
            
            # Vẽ box
            cv2.rectangle(
                annotated,
                (box.x1, box.y1),
                (box.x2, box.y2),
                box_color,
                2
            )
            
            # Vẽ nền cho nhãn
            label = f"{box.class_name}: {box.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(
                annotated,
                (box.x1, box.y1 - label_size[1] - 10),
                (box.x1 + label_size[0] + 5, box.y1),
                box_color,
                -1
            )
            
            # Vẽ text nhãn
            cv2.putText(
                annotated, label,
                (box.x1 + 2, box.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return annotated
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """Chuyển đổi ảnh OpenCV thành chuỗi base64."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')
