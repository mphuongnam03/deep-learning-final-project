import os
img_dir = 'D:/Workspace/ai4life/yolo-dataset/images/val'
label_dir = 'D:/Workspace/ai4life/yolo-dataset/labels/val'
imgs = set(os.listdir(img_dir))
labels = set(os.listdir(label_dir))
print("Số ảnh:", len(imgs))
print("Số label:", len(labels))
print("Ảnh không có label:", imgs - {l.replace('.txt', '.png') for l in labels})
print("Label không có ảnh:", labels - {i.replace('.png', '.txt') for i in imgs})
for fname in os.listdir(label_dir):
    try:
        with open(os.path.join(label_dir, fname), encoding='utf-8') as f:
            f.read()
    except Exception as e:
        print(f"Lỗi encoding: {fname} - {e}")