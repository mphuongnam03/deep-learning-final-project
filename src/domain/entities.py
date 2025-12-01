"""
Tầng Domain - Các Thực Thể Nghiệp Vụ Cốt Lõi

Module này định nghĩa các cấu trúc dữ liệu cốt lõi đại diện cho
các khái niệm cơ bản trong hệ thống chẩn đoán bệnh Lao Phổi (TB).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import time


class DiagnosisClass(str, Enum):
    """Enum đại diện cho 4 lớp chẩn đoán có thể có."""
    HEALTHY = "healthy"
    SICK_BUT_NO_TB = "sick_but_no_tb"
    ACTIVE_TB = "active_tb"
    LATENT_TB = "latent_tb"
    
    @classmethod
    def requires_detection(cls, diagnosis_class: "DiagnosisClass") -> bool:
        """
        Xác định xem lớp chẩn đoán có cần phát hiện bounding box không.
        Chỉ các trường hợp TB (active/latent) mới cần định vị tổn thương.
        """
        return diagnosis_class in [cls.ACTIVE_TB, cls.LATENT_TB]


class BoundingBox(BaseModel):
    """Đại diện cho vùng tổn thương được phát hiện trong ảnh."""
    x1: int = Field(..., description="Tọa độ x góc trên-trái")
    y1: int = Field(..., description="Tọa độ y góc trên-trái")
    x2: int = Field(..., description="Tọa độ x góc dưới-phải")
    y2: int = Field(..., description="Tọa độ y góc dưới-phải")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy phát hiện")
    class_name: str = Field(..., description="Tên lớp phát hiện (active_tb hoặc latent_tb)")
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_dict(self) -> dict:
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": round(self.confidence, 4),
            "class_name": self.class_name,
            "width": self.width,
            "height": self.height
        }


class ClassificationResult(BaseModel):
    """Kết quả từ Giai đoạn 1: Phân loại."""
    predicted_class: DiagnosisClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_probabilities: Optional[dict] = Field(default=None, description="Xác suất cho tất cả các lớp")
    
    
class DetectionResult(BaseModel):
    """Kết quả từ Giai đoạn 2: Phát hiện tổn thương (chỉ cho trường hợp TB)."""
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)
    num_lesions: int = Field(default=0)
    

class DiagnosisResult(BaseModel):
    """
    Kết quả tổng hợp cuối cùng từ Pipeline Cascade 2 Giai Đoạn.
    Đây là thực thể đầu ra chính của hệ thống.
    """
    # Thông tin ảnh đầu vào
    filename: str
    image_width: int
    image_height: int
    
    # Giai đoạn 1: Kết quả phân loại
    predicted_class: DiagnosisClass
    classification_confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: Optional[dict] = None
    
    # Giai đoạn 2: Kết quả phát hiện (chỉ có cho trường hợp TB)
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)
    num_lesions: int = 0
    
    # Thông tin xử lý
    processing_time_ms: float = Field(..., description="Tổng thời gian xử lý (mili giây)")
    stage1_time_ms: float = 0.0
    stage2_time_ms: float = 0.0
    detection_performed: bool = False
    
    # Tùy chọn: Ảnh đã được chú thích ở dạng base64
    annotated_image_base64: Optional[str] = None
    
    def to_json(self) -> dict:
        """Chuyển đổi thành dictionary có thể serialize sang JSON."""
        result = {
            "filename": self.filename,
            "image_size": {
                "width": self.image_width,
                "height": self.image_height
            },
            "diagnosis": {
                "predicted_class": self.predicted_class.value,
                "confidence": round(self.classification_confidence, 4),
                "is_tb_positive": self.predicted_class in [DiagnosisClass.ACTIVE_TB, DiagnosisClass.LATENT_TB]
            },
            "lesions": {
                "detection_performed": self.detection_performed,
                "num_lesions": self.num_lesions,
                "bounding_boxes": [box.to_dict() for box in self.bounding_boxes]
            },
            "processing": {
                "total_time_ms": round(self.processing_time_ms, 2),
                "classification_time_ms": round(self.stage1_time_ms, 2),
                "detection_time_ms": round(self.stage2_time_ms, 2)
            }
        }
        
        if self.class_probabilities:
            result["diagnosis"]["all_probabilities"] = self.class_probabilities
            
        if self.annotated_image_base64:
            result["annotated_image"] = self.annotated_image_base64
            
        return result


@dataclass
class ProcessingContext:
    """
    Đối tượng ngữ cảnh để theo dõi thời gian và trạng thái trong pipeline.
    Hữu ích cho việc debug và giám sát hiệu suất.
    """
    start_time: float = field(default_factory=time.perf_counter)
    stage1_start: Optional[float] = None
    stage1_end: Optional[float] = None
    stage2_start: Optional[float] = None
    stage2_end: Optional[float] = None
    
    def start_classification(self):
        self.stage1_start = time.perf_counter()
        
    def end_classification(self):
        self.stage1_end = time.perf_counter()
        
    def start_detection(self):
        self.stage2_start = time.perf_counter()
        
    def end_detection(self):
        self.stage2_end = time.perf_counter()
        
    @property
    def stage1_time_ms(self) -> float:
        if self.stage1_start and self.stage1_end:
            return (self.stage1_end - self.stage1_start) * 1000
        return 0.0
    
    @property
    def stage2_time_ms(self) -> float:
        if self.stage2_start and self.stage2_end:
            return (self.stage2_end - self.stage2_start) * 1000
        return 0.0
    
    @property
    def total_time_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000
