from ultralytics import YOLO

# Cargar modelo YOLOv8 base
model = YOLO('yolov8n.pt')

# Entrenar
model.train(
    data='C:/Users/GlobalNet/Desktop/placas_YOLOv8/datasets/Placas_Bolivia/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='modeloEntrenado'
)
