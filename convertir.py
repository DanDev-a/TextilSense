from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO('runs/detect/train/weights/best.pt')  # Asegúrate de que la ruta sea correcta

# Exportar a TensorFlow Lite
model.export(format='tflite', imgsz=320)  # Puedes ajustar imgsz si lo necesitas
