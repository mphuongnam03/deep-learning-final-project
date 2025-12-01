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

# Load models globally
print("Loading models...")
cls_model = YOLO(CLS_MODEL_PATH)
det_model = YOLO(DET_MODEL_PATH)
print("✅ Models loaded successfully!")

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    img = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['class']
        conf = det['confidence']
        color = COLORS.get(cls_name, (0, 255, 0))
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label = f"{cls_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - 30), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

def predict(image, conf_threshold):
    """Main prediction function"""
    if image is None:
        return None, "❌ Vui lòng upload ảnh X-quang"
    
    # Ensure image is RGB numpy array
    if isinstance(image, np.ndarray):
        img_rgb = image
    else:
        img_rgb = np.array(image)
    
    # Stage 1: Classification
    cls_result = cls_model.predict(img_rgb, verbose=False)[0]
    cls_probs = cls_result.probs.data.cpu().numpy()
    cls_idx = int(cls_probs.argmax())
    cls_name = CLS_CLASSES[cls_idx]
    cls_conf = float(cls_probs[cls_idx])
    
    # Build result text
    result_text = "## 📋 Kết quả phân loại\n\n"
    result_text += f"**Chẩn đoán:** `{cls_name}`\n\n"
    result_text += f"**Độ tin cậy:** {cls_conf:.1%}\n\n"
    
    # Probabilities table
    result_text += "### Xác suất các lớp:\n\n"
    result_text += "| Lớp | Xác suất |\n|-----|----------|\n"
    for i, prob in enumerate(cls_probs):
        emoji = "✅" if i == cls_idx else ""
        result_text += f"| {CLS_CLASSES[i]} | {prob:.1%} {emoji} |\n"
    
    # Stage 2: Detection (only if TB positive)
    is_tb_positive = cls_name in ["active_tb", "latent_tb"]
    
    detections = []
    if is_tb_positive:
        det_result = det_model.predict(img_rgb, conf=conf_threshold, verbose=False)[0]
        
        for box in det_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            det = {
                "bbox": (x1, y1, x2, y2),
                "class": DET_CLASSES[int(box.cls)],
                "confidence": float(box.conf)
            }
            detections.append(det)
    
    # Detection results
    result_text += "\n## 🔍 Kết quả phát hiện tổn thương\n\n"
    
    if not is_tb_positive:
        result_text += "ℹ️ Không phát hiện lao phổi → Không cần detect vị trí tổn thương\n"
    elif len(detections) == 0:
        result_text += "⚠️ Không phát hiện được tổn thương nào (thử giảm confidence threshold)\n"
    else:
        result_text += f"**Số tổn thương:** {len(detections)}\n\n"
        result_text += "| # | Loại | Độ tin cậy | Vị trí (x1,y1,x2,y2) |\n|---|------|------------|----------------------|\n"
        for i, det in enumerate(detections):
            bbox = det['bbox']
            result_text += f"| {i+1} | {det['class']} | {det['confidence']:.2f} | ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) |\n"
    
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
    
    **Pipeline 2 giai đoạn:**
    1. **Classification:** Phân loại ảnh X-quang thành 4 lớp (healthy, sick_but_no_tb, active_tb, latent_tb)
    2. **Detection:** Nếu phát hiện lao → Xác định vị trí tổn thương
    
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
