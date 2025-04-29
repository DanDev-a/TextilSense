from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
# Convertir el modelo a TFLite

model.eval()
model.model.stride = 32

model.model.input_size = (320, 320)

model.export(format='tflite')

print("Modelo convertido a TFLite con éxito.")