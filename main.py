"""
Điểm Vào Chính - Hệ Thống Chẩn Đoán Bệnh Lao Phổi (TB)

Script này cung cấp CLI thống nhất cho toàn bộ pipeline chẩn đoán TB:
- Tiền xử lý dữ liệu cho cả phân loại và phát hiện
- Huấn luyện model (phân loại và phát hiện)
- Suy luận sử dụng Kiến Trúc Cascade 2 Giai Đoạn
- Đánh giá và trực quan hóa

Kiến trúc: Clean Architecture với dependency injection
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from src packages
from src.data import TBDataPreprocessor
from src.training import TBModelTrainer
from src.evaluation import TBModelEvaluator, TBHeatmapGenerator


def run_inference(args):
    """
    Chạy pipeline suy luận Cascade 2 Giai Đoạn.
    Sử dụng Clean Architecture với các adapter YOLOv8.
    """
    from src.infrastructure import (
        YOLOv8Classifier,
        YOLOv8Detector,
        OpenCVImageLoader,
        TBAnnotator
    )
    from src.usecases import TBDiagnosisUseCase, DiagnosisResultExporter
    
    print("\n" + "=" * 50)
    print("🩺 CHẨN ĐOÁN LPHAU PHỔI - SUY LUẬN CASCADE 2 GIAI ĐOẠN")
    print("=" * 50)
    
    # Khởi tạo các thành phần hạ tầng (Dependency Injection)
    print("\n📦 Đang tải các model...")
    
    try:
        classifier = YOLOv8Classifier(model_path=Path(args.cls_model))
        detector = YOLOv8Detector(model_path=Path(args.det_model))
        image_loader = OpenCVImageLoader()
        annotator = TBAnnotator()
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy model: {e}")
        print("   Vui lòng huấn luyện cả hai model phân loại và phát hiện trước.")
        return
    except Exception as e:
        print(f"❌ Không thể tải model: {e}")
        return
    
    # Khởi tạo use case với các dependency đã tiêm
    diagnosis_service = TBDiagnosisUseCase(
        classifier=classifier,
        detector=detector,
        image_loader=image_loader,
        annotator=annotator
    )
    
    # Chạy suy luận
    image_path = args.image
    print(f"\n🔍 Đang xử lý: {image_path}")
    
    try:
        result = diagnosis_service.diagnose(
            image_path=image_path,
            detection_conf_threshold=args.conf,
            include_annotated_image=args.save_annotated,
            include_probabilities=True
        )
        
        # In tóm tắt
        summary = DiagnosisResultExporter.to_summary(result)
        print(summary)
        
        # Lưu output JSON
        if args.output_json:
            DiagnosisResultExporter.to_json_file(result, args.output_json)
        
        # Lưu ảnh đã chú thích
        if args.save_annotated and result.annotated_image_base64:
            import base64
            output_path = f"annotated_{Path(image_path).name}"
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(result.annotated_image_base64))
            print(f"✅ Ảnh chú thích đã lưu: {output_path}")
            
    except Exception as e:
        print(f"❌ Suy luận thất bại: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Hệ Thống Chẩn Đoán TB - Pipeline Huấn Luyện & Suy Luận',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Tiền xử lý dữ liệu cho huấn luyện
  python main.py --mode preprocess --csv data.csv --images images/ --output dataset/

  # Huấn luyện model phân loại
  python main.py --mode train-cls --output dataset/ --epochs 100

  # Huấn luyện model phát hiện
  python main.py --mode train-det --output dataset/ --epochs 100

  # Chạy suy luận trên một ảnh
  python main.py --mode inference --image test.png --cls-model cls_best.pt --det-model det_best.pt

  # Đánh giá model
  python main.py --mode evaluate --model best.pt --output dataset/
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['preprocess', 'train', 'train-cls', 'train-det', 
                               'evaluate', 'heatmap', 'inference', 'all'],
                       help='Chế độ hoạt động')
    
    # Các tham số dữ liệu
    parser.add_argument('--csv', type=str, default='data.csv',
                       help='Đường dẫn đến file data.csv')
    parser.add_argument('--images', type=str, default='images',
                       help='Thư mục chứa ảnh')
    parser.add_argument('--output', type=str, default='tbx11k-simplified',
                       help='Thư mục đầu ra')
    
    # Các tham số model
    parser.add_argument('--model', type=str, default='best.pt',
                       help='Đường dẫn model cho đánh giá/heatmap')
    parser.add_argument('--cls-model', type=str, default='cls_best.pt',
                       help='Đường dẫn model phân loại (cho suy luận)')
    parser.add_argument('--det-model', type=str, default='det_best.pt',
                       help='Đường dẫn model phát hiện (cho suy luận)')
    
    # Các tham số huấn luyện
    parser.add_argument('--epochs', type=int, default=100,
                       help='Số epochs huấn luyện')
    parser.add_argument('--batch', type=int, default=16,
                       help='Kích thước batch')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Kích thước ảnh')
    
    # Các tham số suy luận
    parser.add_argument('--image', type=str, default=None,
                       help='Đường dẫn ảnh cho suy luận')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Ngưỡng tin cậy cho phát hiện')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Đường dẫn file JSON đầu ra')
    parser.add_argument('--save-annotated', action='store_true',
                       help='Lưu ảnh đã chú thích')
    
    args = parser.parse_args()
    
    # Xử lý chế độ suy luận
    if args.mode == 'inference':
        if not args.image:
            print("❌ Lỗi: --image là bắt buộc cho chế độ suy luận")
            return
        run_inference(args)
        return
    
    # Chế độ tiền xử lý
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n🔄 BƯỚC 1: TIỀN XỬ LÝ")
        print("   Tạo dataset cho cả Phân loại và Phát hiện...")
        preprocessor = TBDataPreprocessor(args.csv, args.images, args.output)
        yaml_path = preprocessor.run()
    
    # Huấn luyện model Phân loại (YOLOv8-cls)
    if args.mode == 'train-cls':
        print("\n🔄 HUẤN LUYỆN: MODEL PHÂN LOẠI (YOLOv8-cls)")
        from ultralytics import YOLO
        
        # YOLOv8-cls yêu cầu cấu trúc thư mục: train/tên_lớp/các_ảnh
        cls_data_path = f"{args.output}/classification"
        if not os.path.exists(cls_data_path):
            print(f"❌ Không tìm thấy dataset phân loại: {cls_data_path}")
            print("   Chạy --mode preprocess trước để tạo dataset.")
            return
            
        model = YOLO('yolov8n-cls.pt')
        results = model.train(
            data=cls_data_path,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch,
            project='tb_classification',
            name='yolov8n_cls',
            exist_ok=True,
            patience=20,
            save=True,
            plots=True,
            verbose=True
        )
        print(f"\n✅ Model phân loại đã lưu vào: {results.save_dir}/weights/best.pt")
    
    # Huấn luyện model Phát hiện (YOLOv8-det)
    if args.mode == 'train-det' or args.mode == 'train':
        print("\n🔄 HUẤN LUYỆN: MODEL PHÁT HIỆN (YOLOv8)")
        yaml_path = f"{args.output}/detection/dataset.yaml"
        if not os.path.exists(yaml_path):
            yaml_path = f"{args.output}/dataset.yaml"
            
        if not os.path.exists(yaml_path):
            print(f"❌ Không tìm thấy dataset phát hiện: {yaml_path}")
            print("   Chạy --mode preprocess trước để tạo dataset.")
            return
            
        trainer = TBModelTrainer(yaml_path, model_size='n')
        results = trainer.train(
            epochs=args.epochs, 
            batch_size=args.batch,
            img_size=args.img_size
        )
        print(f"\n✅ Model phát hiện đã lưu vào: {results.save_dir}/weights/best.pt")
    
    # Huấn luyện cả hai model tuần tự
    if args.mode == 'all':
        print("\n🔄 BƯỚC 2: HUẤN LUYỆN MODEL PHÂN LOẠI")
        # Sẽ gọi logic train-cls ở đây
        
        print("\n🔄 BƯỚC 3: HUẤN LUYỆN MODEL PHÁT HIỆN")
        yaml_path = f"{args.output}/dataset.yaml"
        if os.path.exists(yaml_path):
            trainer = TBModelTrainer(yaml_path, model_size='n')
            results = trainer.train(
                epochs=args.epochs, 
                batch_size=args.batch,
                img_size=args.img_size
            )
            print(f"\n✅ Model phát hiện đã lưu vào: {results.save_dir}/weights/best.pt")
    
    # Chế độ đánh giá
    if args.mode == 'evaluate':
        print("\n🔄 ĐÁNH GIÁ")
        yaml_path = f"{args.output}/dataset.yaml"
        evaluator = TBModelEvaluator(args.model)
        evaluator.evaluate_on_validation(yaml_path)
    
    # Chế độ tạo heatmap
    if args.mode == 'heatmap':
        print("\n🔄 TẠO HEATMAP")
        generator = TBHeatmapGenerator(args.model)
        generator.generate_batch_heatmaps('test', 'test_heatmaps')


if __name__ == '__main__':
    main()