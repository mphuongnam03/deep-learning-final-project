def create_yolo_config(output_dir):
    """
    Tạo file dataset.yaml cho YOLOv8
    """
    config = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 3,  # Số lượng classes
        'names': ['healthy', 'sick_but_no_tb', 'tb']
    }
    
    with open(f'{output_dir}/dataset.yaml', 'w') as f:
        for key, value in config.items():
            if isinstance(value, list):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: '{value}'\n")
    
    print(f"✅ Đã tạo file config: {output_dir}/dataset.yaml")