import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import HTTPException, UploadFile, status

from backend.app.core.config import Settings


TB_CLASSES = {"active_tb", "latent_tb"}
NON_TB_CLASSES = {"healthy", "sick_but_no_tb"}
FALLBACK_CLS_CLASSES = ["active_tb", "healthy", "latent_tb", "sick_but_no_tb"]
FALLBACK_DET_CLASSES = ["active_tb", "latent_tb"]
COLORS_RGB = {
    "active_tb": (220, 38, 38),
    "latent_tb": (245, 158, 11),
}


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for image decoding and annotation") from exc
    return cv2


@dataclass
class LoadedModels:
    cls_model: Any
    det_model: Any
    cls_classes: list[str]
    det_classes: list[str]


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: Optional[LoadedModels] = None

    def load(self) -> LoadedModels:
        if self._models is not None:
            return self._models

        from ultralytics import YOLO

        cls_path = self.settings.resolve_path(self.settings.classification_model_path)
        det_path = self.settings.resolve_path(self.settings.detection_model_path)
        if not cls_path.exists():
            raise FileNotFoundError(f"Classification model not found: {cls_path}")
        if not det_path.exists():
            raise FileNotFoundError(f"Detection model not found: {det_path}")

        cls_model = YOLO(str(cls_path))
        det_model = YOLO(str(det_path))
        self._models = LoadedModels(
            cls_model=cls_model,
            det_model=det_model,
            cls_classes=_extract_names(cls_model, FALLBACK_CLS_CLASSES),
            det_classes=_extract_names(det_model, FALLBACK_DET_CLASSES),
        )
        return self._models

    def status(self) -> dict[str, Any]:
        try:
            models = self.load()
            return {
                "loaded": True,
                "classification_classes": models.cls_classes,
                "detection_classes": models.det_classes,
            }
        except Exception as exc:
            return {"loaded": False, "error": str(exc)}


def _extract_names(model: Any, fallback: list[str]) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return fallback


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


async def decode_upload_image(file: UploadFile) -> np.ndarray:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be an image")
    raw = await file.read()
    return decode_image_bytes(raw)


def decode_image_bytes(raw: bytes) -> np.ndarray:
    cv2 = _cv2()
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty")
    buffer = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not decode uploaded image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict_image(
    img_rgb: np.ndarray,
    filename: str,
    conf_threshold: float,
    models: LoadedModels,
    settings: Settings,
) -> dict[str, Any]:
    start = time.perf_counter()
    img_rgb = _normalize_rgb(img_rgb)
    h, w = img_rgb.shape[:2]

    det_result = models.det_model.predict(img_rgb, conf=conf_threshold, verbose=False)[0]
    raw_detections: list[dict[str, Any]] = []
    for box in det_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        det_class = models.det_classes[class_id] if class_id < len(models.det_classes) else f"class_{class_id}"
        raw_detections.append(
            {
                "bbox": _clip_bbox(x1, y1, x2, y2, w=w, h=h),
                "det_class": det_class,
                "det_conf": float(box.conf[0].cpu().numpy()),
            }
        )

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    crops: list[np.ndarray] = []
    crop_meta: list[dict[str, Any]] = []
    for det in raw_detections:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = (x2 - x1), (y2 - y1)
        if bw < 5 or bh < 5:
            dropped.append({**det, "roi_class": None, "roi_conf": None, "kept": False, "reason": "bbox_too_small"})
            continue

        pad_x = int(bw * settings.roi_padding_ratio)
        pad_y = int(bh * settings.roi_padding_ratio)
        rx1, ry1, rx2, ry2 = _clip_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w=w, h=h)
        crops.append(img_rgb[ry1:ry2, rx1:rx2])
        crop_meta.append({**det, "roi_bbox": (rx1, ry1, rx2, ry2)})

    if crops:
        cls_results = models.cls_model.predict(crops, verbose=False)
        for det, result in zip(crop_meta, cls_results):
            probs = result.probs.data.cpu().numpy()
            idx = int(probs.argmax())
            roi_class = models.cls_classes[idx] if idx < len(models.cls_classes) else f"class_{idx}"
            roi_conf = float(probs[idx])
            enriched = {
                "bbox": det["bbox"],
                "det_class": det["det_class"],
                "det_conf": det["det_conf"],
                "roi_class": roi_class,
                "roi_conf": roi_conf,
                "roi_probs": probs,
            }
            if roi_class in NON_TB_CLASSES and roi_conf >= settings.non_tb_drop_threshold:
                dropped.append({**enriched, "kept": False, "reason": "roi_non_tb_confident"})
            else:
                kept.append({**enriched, "kept": True, "reason": None})

    if kept:
        cls_probs = np.mean(np.stack([item["roi_probs"] for item in kept], axis=0), axis=0)
        cls_source = "roi_filtered_avg"
    else:
        cls_result = models.cls_model.predict(img_rgb, verbose=False)[0]
        cls_probs = cls_result.probs.data.cpu().numpy()
        cls_source = "full_image"

    cls_idx = int(cls_probs.argmax())
    predicted_class = models.cls_classes[cls_idx] if cls_idx < len(models.cls_classes) else f"class_{cls_idx}"
    confidence = float(cls_probs[cls_idx])
    probabilities = [
        {
            "class_name": models.cls_classes[i] if i < len(models.cls_classes) else f"class_{i}",
            "probability": float(prob),
        }
        for i, prob in enumerate(cls_probs)
    ]

    all_boxes = [
        _box_payload(item, kept=True) for item in kept
    ] + [
        _box_payload(item, kept=False) for item in dropped
    ]

    annotated = draw_prediction(img_rgb, predicted_class, confidence, kept)
    processing_time_ms = (time.perf_counter() - start) * 1000
    return {
        "filename": Path(filename).name,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "conf_threshold": conf_threshold,
        "cls_source": cls_source,
        "raw_detection_count": len(raw_detections),
        "kept_detection_count": len(kept),
        "dropped_detection_count": len(dropped),
        "image_width": w,
        "image_height": h,
        "processing_time_ms": processing_time_ms,
        "probabilities": probabilities,
        "boxes": all_boxes,
        "annotated_image_base64": encode_rgb_jpeg_base64(annotated),
    }


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _box_payload(item: dict[str, Any], kept: bool) -> dict[str, Any]:
    return {
        "bbox": [int(v) for v in item["bbox"]],
        "det_class": item["det_class"],
        "det_conf": float(item["det_conf"]),
        "roi_class": item.get("roi_class"),
        "roi_conf": None if item.get("roi_conf") is None else float(item["roi_conf"]),
        "kept": kept,
        "reason": item.get("reason"),
    }


def draw_prediction(
    img_rgb: np.ndarray,
    predicted_class: str,
    confidence: float,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    cv2 = _cv2()
    result = img_rgb.copy()
    h, w = result.shape[:2]
    cv2.rectangle(result, (0, 0), (w, 48), (24, 32, 48), -1)
    if predicted_class == "healthy":
        status_color = (34, 197, 94)
        status_text = "HEALTHY"
    elif predicted_class == "sick_but_no_tb":
        status_color = (234, 179, 8)
        status_text = "SICK (NO TB)"
    else:
        status_color = (239, 68, 68)
        status_text = f"TB DETECTED: {predicted_class.upper()}"
    cv2.putText(result, status_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(result, f"Conf: {confidence:.1%}", (max(12, w - 170), 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        roi_class = det.get("roi_class") or det["det_class"]
        color = COLORS_RGB.get(roi_class, (59, 130, 246))
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        label = f"{roi_class}: {det.get('roi_conf', det['det_conf']):.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        top = max(0, y1 - label_h - 12)
        cv2.rectangle(result, (x1, top), (min(w, x1 + label_w + 10), y1), color, -1)
        cv2.putText(result, label, (x1 + 5, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return result


def encode_rgb_jpeg_base64(img_rgb: np.ndarray) -> str:
    cv2 = _cv2()
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode annotated image")
    return base64.b64encode(buffer).decode("ascii")
