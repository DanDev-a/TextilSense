from ultralytics import YOLO

model = YOLO('yolov8x-seg.pt')

# Entrenar el modelo
model.train(
    data='dataset.yaml',
    epochs=60,
    imgsz=640,
    batch=8,
    name='prueba1',
)