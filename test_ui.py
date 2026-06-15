"""
TB Detection - Visual Test UI với Gradio
Giao diện trực quan để kiểm tra model Classification + Detection
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import gradio as gr

# Paths
DET_MODEL_PATH = "tb_detection/stage2_det/weights/best.pt"
CLS_MODEL_PATH = "tb_classification/stage1_cls/weights/best.pt"

# Class names
CLS_CLASSES = ["active_tb", "healthy", "latent_tb", "sick_but_no_tb"]
DET_CLASSES = ["active_tb", "latent_tb"]

# Colors for detection boxes (RGB for Gradio)
COLORS = {
    "active_tb": (255, 0, 0),      # Red
    "latent_tb": (255, 165, 0),    # Orange
}

# Drop bbox only if ROI classification is confidently non-TB
NON_TB_DROP_THRESHOLD = 0.80

# Load models globally
print("Loading models...")
cls_model = YOLO(CLS_MODEL_PATH)
det_model = YOLO(DET_MODEL_PATH)
print("✅ Models loaded successfully!")

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    img = np.ascontiguousarray(image.copy())
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['class']
        conf = det['confidence']
        # OpenCV expects BGR; our COLORS are RGB
        color_rgb = COLORS.get(cls_name, (0, 255, 0))
        color = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

        h, w = img.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(int(x1), int(y1), int(x2), int(y2), w=w, h=h)
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label = f"{cls_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - 30), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _classify_from_detections(img_rgb: np.ndarray, detections: list):
    """Run classification on detected ROIs; fallback to full image if no valid crops."""
    h, w = img_rgb.shape[:2]
    crops = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1, y1, x2, y2 = _clip_bbox(int(x1), int(y1), int(x2), int(y2), w=w, h=h)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue
        crops.append(img_rgb[y1:y2, x1:x2])

    # If no usable crops, classify the full image
    if not crops:
        cls_result = cls_model.predict(img_rgb, verbose=False)[0]
        cls_probs = cls_result.probs.data.cpu().numpy()
        return cls_probs, "full_image"

    # Ultralytics supports batch predict with list/array of images
    results = cls_model.predict(crops, verbose=False)
    probs_list = [r.probs.data.cpu().numpy() for r in results]
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    return avg_probs, "roi_batch"

def predict(image, conf_threshold):
    """Main prediction function"""
    if image is None:
        return None, "❌ Vui lòng upload ảnh X-quang"
    
    # Ensure image is RGB numpy array
    if isinstance(image, np.ndarray):
        img_rgb = image
    else:
        img_rgb = np.array(image)

    # Normalize to 3-channel RGB
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb, img_rgb, img_rgb], axis=-1)
    elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        img_rgb = img_rgb[:, :, :3]

    # Ensure dtype for OpenCV drawing
    img_rgb = np.ascontiguousarray(img_rgb)
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    
    # Stage 1: Detection (run first on full image)
    raw_detections = []
    det_result = det_model.predict(img_rgb, conf=conf_threshold, verbose=False)[0]
    for box in det_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        raw_detections.append({
            "bbox": (x1, y1, x2, y2),
            "det_class": DET_CLASSES[int(box.cls)],
            "det_conf": float(box.conf),
        })

    # Stage 2: Classification per-bbox, and drop non-TB boxes
    # Rule: if ROI classification is NOT TB -> remove that bbox from detections.
    tb_classes = {"active_tb", "latent_tb"}
    non_tb_classes = {"healthy", "sick_but_no_tb"}

    detections = []
    dropped = []
    if raw_detections:
        h, w = img_rgb.shape[:2]
        crops = []
        crop_meta = []
        for det in raw_detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = _clip_bbox(int(x1), int(y1), int(x2), int(y2), w=w, h=h)
            bw, bh = (x2 - x1), (y2 - y1)
            if bw < 5 or bh < 5:
                dropped.append({
                    **det,
                    "reason": "bbox_too_small",
                })
                continue

            # Add padding for ROI classification to include more context
            pad_ratio = 0.20
            pad_x = int(bw * pad_ratio)
            pad_y = int(bh * pad_ratio)
            rx1, ry1, rx2, ry2 = _clip_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w=w, h=h)

            crops.append(img_rgb[ry1:ry2, rx1:rx2])
            crop_meta.append({
                **det,
                # Keep draw bbox (tight) and ROI bbox (padded) separately
                "bbox": (x1, y1, x2, y2),
                "roi_bbox": (rx1, ry1, rx2, ry2),
            })

        if crops:
            cls_results = cls_model.predict(crops, verbose=False)
            for det, r in zip(crop_meta, cls_results):
                probs = r.probs.data.cpu().numpy()
                idx = int(probs.argmax())
                roi_cls = CLS_CLASSES[idx]
                roi_conf = float(probs[idx])

                det_enriched = {
                    "bbox": det["bbox"],
                    # Use ROI label when it's TB; otherwise keep det label for drawing
                    "class": roi_cls if roi_cls in tb_classes else det["det_class"],
                    "confidence": det["det_conf"],
                    "det_class": det["det_class"],
                    "det_conf": det["det_conf"],
                    "roi_cls": roi_cls,
                    "roi_conf": roi_conf,
                    "roi_probs": probs,
                }

                if roi_cls in non_tb_classes and roi_conf >= NON_TB_DROP_THRESHOLD:
                    det_enriched["reason"] = "roi_non_tb_confident"
                    dropped.append(det_enriched)
                else:
                    # Keep boxes unless we're confidently sure they're non-TB.
                    detections.append(det_enriched)
        else:
            # All boxes were invalid/tiny; fall back to full image classification.
            pass

    # Aggregate classification: average probs over remaining TB ROIs; fallback to full image
    if detections:
        cls_probs = np.mean(np.stack([d["roi_probs"] for d in detections], axis=0), axis=0)
        cls_source = "roi_filtered_avg"
    else:
        cls_result = cls_model.predict(img_rgb, verbose=False)[0]
        cls_probs = cls_result.probs.data.cpu().numpy()
        cls_source = "full_image"

    cls_idx = int(cls_probs.argmax())
    cls_name = CLS_CLASSES[cls_idx]
    cls_conf = float(cls_probs[cls_idx])

    # Build result text
    result_text = "## 🔍 Kết quả phát hiện tổn thương\n\n"
    result_text += f"**Confidence threshold:** {conf_threshold:.2f}\n\n"
    result_text += f"**BBox detect được:** {len(raw_detections)}\n\n"
    if dropped:
        result_text += f"**BBox bị loại (do classification/non-TB):** {len(dropped)}\n\n"

    if len(raw_detections) == 0:
        result_text += "⚠️ Không phát hiện được bbox nào (thử giảm confidence threshold)\n"
    elif len(detections) == 0:
        result_text += "ℹ️ Có bbox nhưng classification cho rằng không phải TB → đã loại hết bbox\n"
    else:
        result_text += f"**BBox giữ lại (TB):** {len(detections)}\n\n"
        result_text += "| # | ROI Class | ROI Conf | Det Class | Det Conf | Vị trí (x1,y1,x2,y2) |\n|---|----------|----------|----------|----------|----------------------|\n"
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            result_text += (
                f"| {i+1} | {det['roi_cls']} | {det['roi_conf']:.2f} | {det['det_class']} | {det['det_conf']:.2f} | "
                f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) |\n"
            )

    result_text += "\n## 📋 Kết quả phân loại\n\n"
    result_text += f"**Chẩn đoán:** `{cls_name}`\n\n"
    result_text += f"**Độ tin cậy:** {cls_conf:.1%}\n\n"
    result_text += f"**Nguồn classification:** `{cls_source}`\n\n"

    # Probabilities table
    result_text += "### Xác suất các lớp:\n\n"
    result_text += "| Lớp | Xác suất |\n|-----|----------|\n"
    for i, prob in enumerate(cls_probs):
        emoji = "✅" if i == cls_idx else ""
        result_text += f"| {CLS_CLASSES[i]} | {prob:.1%} {emoji} |\n"
    
    # Draw results on image
    result_img = img_rgb.copy()
    
    # Draw header
    h, w = result_img.shape[:2]
    cv2.rectangle(result_img, (0, 0), (w, 45), (50, 50, 50), -1)
    
    # Status color
    if cls_name == "healthy":
        status_color = (0, 255, 0)  # Green
        status_text = "HEALTHY"
    elif cls_name == "sick_but_no_tb":
        status_color = (255, 255, 0)  # Yellow
        status_text = "SICK (NO TB)"
    else:
        status_color = (255, 0, 0)  # Red
        status_text = f"TB DETECTED: {cls_name.upper()}"
    
    cv2.putText(result_img, status_text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    cv2.putText(result_img, f"Conf: {cls_conf:.1%}", (w - 150, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw detections
    if detections:
        result_img = draw_detections(result_img, detections)
    
    return result_img, result_text

def load_sample_image(sample_type):
    """Load sample image from validation set"""
    val_dir = Path("datasets/dataset_det/images/val")
    
    if sample_type == "TB Active":
        prefix = "tb"
    elif sample_type == "TB Latent":
        prefix = "tb"  
    elif sample_type == "Healthy":
        prefix = "h"
    else:  # Sick but no TB
        prefix = "s"
    
    images = list(val_dir.glob(f"{prefix}*.png"))
    if images:
        import random
        img_path = random.choice(images)
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    return None

# Create Gradio Interface
with gr.Blocks(title="TB Detection System") as demo:
    gr.Markdown("""
    # 🫁 Hệ thống Phát hiện Lao phổi (TB Detection)
    
    **Pipeline 2 giai đoạn (đã đổi thứ tự):**
    1. **Detection:** Phát hiện vùng tổn thương (bbox)
    2. **Classification:** Phân loại (ưu tiên dựa trên các vùng bbox; nếu không có bbox thì phân loại toàn ảnh)
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload ảnh X-quang")
            input_image = gr.Image(label="Ảnh X-quang ngực", type="numpy")
            
            conf_slider = gr.Slider(
                minimum=0.1, 
                maximum=0.9, 
                value=0.25, 
                step=0.05,
                label="Detection Confidence Threshold"
            )
            
            with gr.Row():
                predict_btn = gr.Button("🔍 Phân tích", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Xóa", size="lg")
            
            gr.Markdown("### 📁 Ảnh mẫu")
            with gr.Row():
                sample_tb = gr.Button("🔴 TB Sample", size="sm")
                sample_healthy = gr.Button("🟢 Healthy", size="sm")
                sample_sick = gr.Button("🟡 Sick (no TB)", size="sm")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Kết quả")
            output_image = gr.Image(label="Ảnh với Bounding Box", type="numpy")
            
    with gr.Row():
        output_text = gr.Markdown(label="Chi tiết kết quả")
    
    gr.Markdown("""
    ---
    ### 📊 Thông tin Model
    
    | Model | Accuracy/mAP | Thông số |
    |-------|-------------|----------|
    | Classification (YOLOv8n-cls) | 98.1% | 4 classes |
    | Detection (YOLOv8n) | 43% mAP50, 66% Recall | 2 classes |
    
    **Chú thích màu:**
    - 🔴 **Đỏ:** Active TB
    - 🟠 **Cam:** Latent TB
    """)
    
    # Event handlers
    predict_btn.click(
        fn=predict,
        inputs=[input_image, conf_slider],
        outputs=[output_image, output_text]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, ""),
        outputs=[input_image, output_image, output_text]
    )
    
    sample_tb.click(
        fn=lambda: load_sample_image("TB Active"),
        outputs=[input_image]
    )
    
    sample_healthy.click(
        fn=lambda: load_sample_image("Healthy"),
        outputs=[input_image]
    )
    
    sample_sick.click(
        fn=lambda: load_sample_image("Sick but no TB"),
        outputs=[input_image]
    )

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting TB Detection UI...")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
