"""
Script đơn giản để chạy phân tích dữ liệu
Chỉ phân tích và hiển thị thống kê, không chuyển đổi dữ liệu
"""

import argparse
from preprocessing import TBDataPreprocessor

def main():
    parser = argparse.ArgumentParser(description='Phân tích dữ liệu TB Detection')
    parser.add_argument('--csv', type=str, default='tbx11k-simplified/data.csv',
                       help='Đường dẫn đến data.csv')
    parser.add_argument('--images', type=str, default='tbx11k-simplified/images',
                       help='Thư mục chứa ảnh (tùy chọn, để kiểm tra ảnh có tồn tại)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📊 PHÂN TÍCH DỮ LIỆU TB DETECTION")
    print("="*60)
    
    # Tạo preprocessor (output_dir không quan trọng vì chỉ phân tích)
    preprocessor = TBDataPreprocessor(
        csv_path=args.csv,
        images_dir=args.images,
        output_dir='temp'  # Tạm thời, không dùng
    )
    
    # Chỉ chạy phân tích, không chuyển đổi
    print(f"\n📖 Đọc file CSV: {args.csv}")
    df = preprocessor.parse_csv()
    
    print("\n" + "="*60)
    preprocessor.analyze_dataset()
    print("="*60)
    
    print("\n✅ Hoàn tất phân tích!")

if __name__ == '__main__':
    main()

