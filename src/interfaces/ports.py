"""
Tầng Interfaces - Các Cổng (Lớp Trừ C Tượng)

Các lớp trừ tượng này định nghĩa các hợp đồng mà các adapter hạ tầng phải triển khai.
Điều này cho phép thay thế YOLO bằng framework khác (ví dụ: Detectron2, MMDetection)
mà không cần thay đổi logic nghiệp vụ.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

from src.domain.entities import (
    ClassificationResult,
    DetectionResult,
    BoundingBox,
    DiagnosisClass
)


class ImageClassifierPort(ABC):
    """
    Interface trừ tượng cho các model phân loại ảnh.
    Giai đoạn 1 của cascade sử dụng interface này để phân loại ảnh X-quang ngực.
    """
    
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Trả về danh sách tên các lớp theo thứ tự chỉ số."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Trả về số lượng lớp."""
        pass
    
    @abstractmethod
    def classify(
        self, 
        image: np.ndarray,
        return_probabilities: bool = False
    ) -> ClassificationResult:
        """
        Phân loại ảnh vào một trong các lớp chẩn đoán.
        
        Args:
            image: Ảnh đầu vào dạng numpy array (định dạng BGR, H x W x C)
            return_probabilities: Nếu True, bao gồm xác suất cho tất cả các lớp
            
        Returns:
            ClassificationResult với lớp dự đoán và độ tin cậy
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """Tải trọng số model từ file."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Kiểm tra xem model đã được tải và sẵn sàng suy luận chưa."""
        pass


class ObjectDetectorPort(ABC):
    """
    Interface trừ tượng cho các model phát hiện đối tượng.
    Giai đoạn 2 của cascade sử dụng interface này để phát hiện vùng tổn thương TB.
    """
    
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Trả về danh sách tên các lớp có thể phát hiện."""
        pass
    
    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> DetectionResult:
        """
        Phát hiện vùng tổn thương TB trong ảnh.
        
        Args:
            image: Ảnh đầu vào dạng numpy array (định dạng BGR, H x W x C)
            confidence_threshold: Ngưỡng tin cậy tối thiểu để giữ lại phát hiện
            iou_threshold: Ngưỡng IoU cho NMS (Non-Maximum Suppression)
            
        Returns:
            DetectionResult với danh sách bounding box
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """Tải trọng số model từ file."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Kiểm tra xem model đã được tải và sẵn sàng suy luận chưa."""
        pass


class ImageAnnotatorPort(ABC):
    """
    Interface trừ tượng để chú thích ảnh với kết quả phát hiện.
    Cho phép thay đổi backend hiển thị nếu cần.
    """
    
    @abstractmethod
    def annotate(
        self,
        image: np.ndarray,
        diagnosis_class: DiagnosisClass,
        bounding_boxes: List[BoundingBox],
        classification_confidence: float
    ) -> np.ndarray:
        """
        Vẽ các chú thích lên ảnh.
        
        Args:
            image: Ảnh gốc
            diagnosis_class: Lớp chẩn đoán dự đoán
            bounding_boxes: Danh sách các vùng tổn thương được phát hiện
            classification_confidence: Độ tin cậy phân loại
            
        Returns:
            Ảnh đã được chú thích dạng numpy array
        """
        pass
    
    @abstractmethod
    def image_to_base64(self, image: np.ndarray) -> str:
        """Chuyển đổi ảnh thành chuỗi base64 để serialize JSON."""
        pass


class ImageLoaderPort(ABC):
    """
    Interface trừ tượng để tải ảnh từ nhiều nguồn khác nhau.
    """
    
    @abstractmethod
    def load(self, source: str) -> Tuple[np.ndarray, int, int]:
        """
        Tải ảnh từ đường dẫn file hoặc URL.
        
        Args:
            source: Đường dẫn file hoặc URL
            
        Returns:
            Tuple gồm (mảng ảnh, chiều rộng, chiều cao)
        """
        pass
    
    @abstractmethod
    def validate(self, source: str) -> bool:
        """Kiểm tra xem nguồn có phải là ảnh hợp lệ không."""
        pass
