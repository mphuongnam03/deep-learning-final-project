"""
Tầng Use Cases - Bộ Điều Khiển Chẩn Đoán TB

Module này chứa logic nghiệp vụ cốt lõi điều phối Pipeline Cascade
2 Giai Đoạn cho chẩn đoán TB. Nó điều phối luồng giữa phân loại
và phát hiện trong khi vẫn độc lập với các triển khai cụ thể.
"""

from pathlib import Path
from typing import Optional
import json

from src.domain.entities import (
    DiagnosisResult,
    DiagnosisClass,
    ProcessingContext,
    BoundingBox
)
from src.interfaces.ports import (
    ImageClassifierPort,
    ObjectDetectorPort,
    ImageAnnotatorPort,
    ImageLoaderPort
)


class TBDiagnosisUseCase:
    """
    Use case chính triển khai Kiến Trúc Cascade 2 Giai Đoạn.
    
    Luồng Pipeline:
    1. Tải và xác thực ảnh đầu vào
    2. Giai đoạn 1: Phân loại ảnh vào 4 lớp (healthy, sick_but_no_tb, active_tb, latent_tb)
    3. Giai đoạn 2 có điều kiện: Nếu TB dương tính (active/latent), chạy phát hiện để định vị tổn thương
    4. Chú thích ảnh và trả về kết quả có cấu trúc
    
    Lớp này chỉ phụ thuộc vào các interface trừ tượng, cho phép các
    triển khai khác nhau được tiêm vào (ví dụ: thay YOLO bằng framework khác).
    """
    
    def __init__(
        self,
        classifier: ImageClassifierPort,
        detector: ObjectDetectorPort,
        image_loader: ImageLoaderPort,
        annotator: Optional[ImageAnnotatorPort] = None
    ):
        self._classifier = classifier
        self._detector = detector
        self._image_loader = image_loader
        self._annotator = annotator
        
        # Xác thực các model đã được tải
        if not classifier.is_loaded():
            raise RuntimeError("Model phân loại chưa được tải")
        if not detector.is_loaded():
            raise RuntimeError("Model phát hiện chưa được tải")
    
    def diagnose(
        self,
        image_path: str,
        detection_conf_threshold: float = 0.25,
        include_annotated_image: bool = True,
        include_probabilities: bool = True
    ) -> DiagnosisResult:
        """
        Chạy pipeline chẩn đoán TB hoàn chỉnh trên một ảnh.
        
        Args:
            image_path: Đường dẫn đến ảnh X-quang ngực
            detection_conf_threshold: Ngưỡng tin cậy cho phát hiện tổn thương
            include_annotated_image: Có bao gồm ảnh đã chú thích dạng base64 trong kết quả không
            include_probabilities: Có bao gồm xác suất của tất cả các lớp không
            
        Returns:
            DiagnosisResult chứa kết quả phân loại, phát hiện (nếu có) và metadata
        """
        context = ProcessingContext()
        
        # Bước 1: Tải và xác thực ảnh
        image, width, height = self._image_loader.load(image_path)
        filename = Path(image_path).name
        
        # Bước 2: Giai đoạn 1 - Phân loại
        context.start_classification()
        classification_result = self._classifier.classify(
            image=image,
            return_probabilities=include_probabilities
        )
        context.end_classification()
        
        predicted_class = classification_result.predicted_class
        classification_confidence = classification_result.confidence
        
        # Bước 3: Giai đoạn 2 có điều kiện - Phát hiện (chỉ cho trường hợp TB dương tính)
        # Đây là logic cascade chính: bỏ qua phát hiện hoàn toàn cho các trường hợp không TB
        # để tiết kiệm tính toán và tránh phát hiện tổn thương sai
        bounding_boxes = []
        detection_performed = False
        
        if DiagnosisClass.requires_detection(predicted_class):
            context.start_detection()
            detection_result = self._detector.detect(
                image=image,
                confidence_threshold=detection_conf_threshold
            )
            context.end_detection()
            
            bounding_boxes = detection_result.bounding_boxes
            detection_performed = True
        
        # Bước 4: Tạo ảnh đã chú thích (nếu được yêu cầu và annotator có sẵn)
        annotated_image_base64 = None
        if include_annotated_image and self._annotator:
            annotated_image = self._annotator.annotate(
                image=image,
                diagnosis_class=predicted_class,
                bounding_boxes=bounding_boxes,
                classification_confidence=classification_confidence
            )
            annotated_image_base64 = self._annotator.image_to_base64(annotated_image)
        
        # Xây dựng kết quả cuối cùng
        return DiagnosisResult(
            filename=filename,
            image_width=width,
            image_height=height,
            predicted_class=predicted_class,
            classification_confidence=classification_confidence,
            class_probabilities=classification_result.all_probabilities,
            bounding_boxes=bounding_boxes,
            num_lesions=len(bounding_boxes),
            processing_time_ms=context.total_time_ms,
            stage1_time_ms=context.stage1_time_ms,
            stage2_time_ms=context.stage2_time_ms,
            detection_performed=detection_performed,
            annotated_image_base64=annotated_image_base64
        )
    
    def diagnose_batch(
        self,
        image_paths: list,
        detection_conf_threshold: float = 0.25,
        include_annotated_image: bool = False
    ) -> list:
        """
        Chạy chẩn đoán trên nhiều ảnh.
        
        Args:
            image_paths: Danh sách đường dẫn đến các ảnh X-quang ngựcuang ngực
            detection_conf_threshold: Ngưỡng tin cậy cho phát hiện tổn thương
            include_annotated_image: Có bao gồm ảnh đã chú thích dạng base64 không
            
        Returns:
            Danh sách các đối tượng DiagnosisResult
        """
        results = []
        for path in image_paths:
            try:
                result = self.diagnose(
                    image_path=str(path),
                    detection_conf_threshold=detection_conf_threshold,
                    include_annotated_image=include_annotated_image
                )
                results.append(result)
            except Exception as e:
                print(f"⚠️ Không thể xử lý {path}: {e}")
                continue
        return results


class DiagnosisResultExporter:
    """
    Lớp tiện ích để xuất kết quả chẩn đoán ra các định dạng khác nhau.
    """
    
    @staticmethod
    def to_json(result: DiagnosisResult, indent: int = 2) -> str:
        """Xuất kết quả thành chuỗi JSON."""
        return json.dumps(result.to_json(), indent=indent, ensure_ascii=False)
    
    @staticmethod
    def to_json_file(result: DiagnosisResult, output_path: str) -> None:
        """Lưu kết quả vào file JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_json(), f, indent=2, ensure_ascii=False)
        print(f"✅ Kết quả đã lưu vào: {output_path}")
    
    @staticmethod
    def to_summary(result: DiagnosisResult) -> str:
        """Tạo tóm tắt dễ đọc cho người dùng."""
        lines = [
            "=" * 50,
            "🩺 TB DIAGNOSIS RESULT",
            "=" * 50,
            f"📁 File: {result.filename}",
            f"📐 Size: {result.image_width}x{result.image_height}",
            "",
            "📊 CLASSIFICATION:",
            f"   Class: {result.predicted_class.value}",
            f"   Confidence: {result.classification_confidence:.1%}",
            f"   TB Positive: {'Yes' if result.predicted_class in [DiagnosisClass.ACTIVE_TB, DiagnosisClass.LATENT_TB] else 'No'}",
        ]
        
        if result.detection_performed:
            lines.extend([
                "",
                "🔍 DETECTION:",
                f"   Lesions found: {result.num_lesions}",
            ])
            for i, box in enumerate(result.bounding_boxes, 1):
                lines.append(f"   [{i}] {box.class_name}: {box.confidence:.2%} at ({box.x1},{box.y1})-({box.x2},{box.y2})")
        
        lines.extend([
            "",
            "⏱️ TIMING:",
            f"   Classification: {result.stage1_time_ms:.1f}ms",
            f"   Detection: {result.stage2_time_ms:.1f}ms" if result.detection_performed else "   Detection: Skipped",
            f"   Total: {result.processing_time_ms:.1f}ms",
            "=" * 50
        ])
        
        return "\n".join(lines)
