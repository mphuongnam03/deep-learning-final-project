# TB Diagnosis System - Clean Architecture

# Domain Layer
from .domain import (
    DiagnosisClass,
    BoundingBox,
    ClassificationResult,
    DetectionResult,
    DiagnosisResult,
    ProcessingContext
)

# Interfaces Layer
from .interfaces import (
    ImageClassifierPort,
    ObjectDetectorPort,
    ImageAnnotatorPort,
    ImageLoaderPort
)

# Infrastructure Layer
from .infrastructure import (
    YOLOv8Classifier,
    YOLOv8Detector,
    OpenCVImageLoader,
    TBAnnotator
)

# Use Cases Layer
from .usecases import (
    TBDiagnosisUseCase,
    DiagnosisResultExporter
)

# Data Processing
from .data import TBDataPreprocessor

# Training
from .training import TBModelTrainer

# Evaluation
from .evaluation import TBModelEvaluator, TBHeatmapGenerator

__all__ = [
    # Domain
    "DiagnosisClass", "BoundingBox", "ClassificationResult", 
    "DetectionResult", "DiagnosisResult", "ProcessingContext",
    # Interfaces
    "ImageClassifierPort", "ObjectDetectorPort", 
    "ImageAnnotatorPort", "ImageLoaderPort",
    # Infrastructure
    "YOLOv8Classifier", "YOLOv8Detector", 
    "OpenCVImageLoader", "TBAnnotator",
    # Use Cases
    "TBDiagnosisUseCase", "DiagnosisResultExporter",
    # Data
    "TBDataPreprocessor",
    # Training
    "TBModelTrainer",
    # Evaluation
    "TBModelEvaluator", "TBHeatmapGenerator"
]
