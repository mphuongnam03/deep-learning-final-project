"""
Script kiểm tra kết quả Detection YOLOv8
Hiển thị ảnh với bounding box và lưu kết quả
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import random

# Paths
DET_MODEL_PATH = "tb_detection/stage2_det/weights/best.pt"
CLS_MODEL_PATH = "tb_classification/stage1_cls/weights/best.pt"

# Test images - sử dụng validation set
TEST_IMAGES_DIR = "datasets/dataset_det/images/val"
OUTPUT_DIR = "test_results"

# Class names
CLS_CLASSES = ["active_tb", "healthy", "latent_tb", "sick_but_no_tb"]
DET_CLASSES = ["active_tb", "latent_tb"]

# Colors for detection boxes (BGR)
COLORS = {
    "active_tb": (0, 0, 255),      # Red
    "latent_tb": (0, 165, 255),    # Orange
}

def load_models():
    """Load Classification and Detection models"""
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)
    
    cls_model = YOLO(CLS_MODEL_PATH)
    det_model = YOLO(DET_MODEL_PATH)
    
    print(f"✅ Classification model: {CLS_MODEL_PATH}")
    print(f"✅ Detection model: {DET_MODEL_PATH}")
    print()
    return cls_model, det_model

def draw_results(image, cls_result, det_results):
    """Draw classification and detection results on image"""
    img = image.copy()
    h, w = img.shape[:2]
    
    # Draw detection boxes
    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_idx = int(box.cls)
        conf = float(box.conf)
        cls_name = DET_CLASSES[cls_idx]
        color = COLORS.get(cls_name, (0, 255, 0))
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{cls_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - 30), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw classification result at top
    cls_probs = cls_result.probs.data.cpu().numpy()
    cls_idx = int(cls_probs.argmax())
    cls_name = CLS_CLASSES[cls_idx]
    cls_conf = float(cls_probs[cls_idx])
    
    # Background for classification text
    cv2.rectangle(img, (0, 0), (w, 40), (50, 50, 50), -1)
    
    # Classification text
    cls_text = f"Classification: {cls_name} ({cls_conf:.1%})"
    cv2.putText(img, cls_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Detection count
    det_count = len(det_results.boxes)
    det_text = f"Detections: {det_count}"
    cv2.putText(img, det_text, (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return img

def test_single_image(cls_model, det_model, image_path, save_path=None):
    """Test pipeline on single image"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load image: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Stage 1: Classification
    cls_result = cls_model.predict(img_rgb, verbose=False)[0]
    cls_probs = cls_result.probs.data.cpu().numpy()
    cls_idx = int(cls_probs.argmax())
    cls_name = CLS_CLASSES[cls_idx]
    cls_conf = float(cls_probs[cls_idx])
    
    print(f"\n📷 Image: {Path(image_path).name}")
    print(f"   Classification: {cls_name} ({cls_conf:.1%})")
    
    # Stage 2: Detection
    det_result = det_model.predict(img_rgb, conf=0.25, verbose=False)[0]
    det_count = len(det_result.boxes)
    
    print(f"   Detections: {det_count} lesions found")
    
    for i, box in enumerate(det_result.boxes):
        det_cls = DET_CLASSES[int(box.cls)]
        det_conf = float(box.conf)
        bbox = box.xyxy[0].cpu().numpy()
        print(f"      [{i+1}] {det_cls}: {det_conf:.2f} - bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    
    # Draw results
    result_img = draw_results(img, cls_result, det_result)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, result_img)
        print(f"   💾 Saved: {save_path}")
    
    return {
        "classification": {"class": cls_name, "confidence": cls_conf},
        "detections": det_count,
        "image": result_img
    }

def test_batch(cls_model, det_model, num_samples=10):
    """Test on random samples from validation set"""
    print("\n" + "=" * 60)
    print(f"Testing on {num_samples} random images from validation set")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all images
    image_files = list(Path(TEST_IMAGES_DIR).glob("*.png")) + \
                  list(Path(TEST_IMAGES_DIR).glob("*.jpg"))
    
    if not image_files:
        print(f"❌ No images found in {TEST_IMAGES_DIR}")
        return
    
    print(f"Found {len(image_files)} images in validation set")
    
    # Random sample
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Test each image
    results = []
    for img_path in samples:
        save_path = os.path.join(OUTPUT_DIR, f"result_{img_path.name}")
        result = test_single_image(cls_model, det_model, str(img_path), save_path)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_detections = sum(r["detections"] for r in results)
    cls_counts = {}
    for r in results:
        cls_name = r["classification"]["class"]
        cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1
    
    print(f"Total images tested: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"\nClassification distribution:")
    for cls_name, count in cls_counts.items():
        print(f"   {cls_name}: {count}")
    
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")

def test_specific_images(cls_model, det_model, image_paths):
    """Test on specific image paths"""
    print("\n" + "=" * 60)
    print("Testing specific images")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            save_path = os.path.join(OUTPUT_DIR, f"result_{Path(img_path).name}")
            test_single_image(cls_model, det_model, img_path, save_path)
        else:
            print(f"❌ Image not found: {img_path}")

def interactive_test(cls_model, det_model):
    """Interactive testing mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Enter image path to test, or 'q' to quit")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    while True:
        img_path = input("\nImage path: ").strip()
        
        if img_path.lower() == 'q':
            print("Goodbye!")
            break
        
        if os.path.exists(img_path):
            save_path = os.path.join(OUTPUT_DIR, f"result_{Path(img_path).name}")
            result = test_single_image(cls_model, det_model, img_path, save_path)
            
            if result:
                # Show image
                cv2.imshow("Result", result["image"])
                print("\nPress any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"❌ File not found: {img_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TB Detection Pipeline")
    parser.add_argument("--mode", type=str, default="batch", 
                        choices=["batch", "interactive", "single"],
                        help="Test mode: batch, interactive, or single")
    parser.add_argument("--image", type=str, help="Image path for single mode")
    parser.add_argument("--num", type=int, default=10, help="Number of samples for batch mode")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    # Load models
    cls_model, det_model = load_models()
    
    if args.mode == "batch":
        test_batch(cls_model, det_model, args.num)
    elif args.mode == "interactive":
        interactive_test(cls_model, det_model)
    elif args.mode == "single" and args.image:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, f"result_{Path(args.image).name}")
        test_single_image(cls_model, det_model, args.image, save_path)
    else:
        print("Usage examples:")
        print("  python test_detection.py --mode batch --num 10")
        print("  python test_detection.py --mode single --image path/to/image.png")
        print("  python test_detection.py --mode interactive")
