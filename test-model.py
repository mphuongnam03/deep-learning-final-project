from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict(source='test_images/', save=True)