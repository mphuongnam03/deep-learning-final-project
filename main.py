"""
Main script để chạy toàn bộ pipeline
"""

import argparse
from preprocessing import TBDataPreprocessor
from train import TBModelTrainer
from evaluate import TBModelEvaluator
from heatmap import TBHeatmapGenerator

def main():
    parser = argparse.ArgumentParser(description='TB Detection Training Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['preprocess', 'train', 'evaluate', 'heatmap', 'all'],
                       help='Chế độ chạy')
    parser.add_argument('--csv', type=str, default='data.csv',
                       help='Đường dẫn đến data.csv')
    parser.add_argument('--images', type=str, default='images',
                       help='Thư mục chứa ảnh')
    parser.add_argument('--output', type=str, default='tbx11k-simplified',
                       help='Thư mục output')
    parser.add_argument('--model', type=str, default='best.pt',
                       help='Đường dẫn đến model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Số epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Image size')
    
    args = parser.parse_args()
    
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n🔄 BƯỚC 1: PREPROCESSING")
        preprocessor = TBDataPreprocessor(args.csv, args.images, args.output)
        yaml_path = preprocessor.run()
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n🔄 BƯỚC 2: TRAINING")
        yaml_path = f"{args.output}/dataset.yaml"
        trainer = TBModelTrainer(yaml_path, model_size='n')
        results = trainer.train(epochs=args.epochs, batch_size=args.batch, 
                               img_size=args.img_size)
        
        print(f"\n✅ Model saved to: {results.save_dir}/weights/best.pt")
    
    if args.mode == 'evaluate':
        print("\n🔄 EVALUATION")
        yaml_path = f"{args.output}/dataset.yaml"
        evaluator = TBModelEvaluator(args.model)
        evaluator.evaluate_on_validation(yaml_path)
    
    if args.mode == 'heatmap':
        print("\n🔄 HEATMAP GENERATION")
        generator = TBHeatmapGenerator(args.model)
        # Ví dụ: tạo heatmap cho ảnh test
        generator.generate_batch_heatmaps('test', 'test_heatmaps')

if __name__ == '__main__':
    main()