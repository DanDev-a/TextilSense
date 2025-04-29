from ultralytics import YOLO

# Cargar modelo previamente entrenado (placas colombianas)
model = YOLO('runs/detect/placas-bolivianas-finetune4/weights/best.pt')  # Ajusta la ruta si es diferente

# Reentrenar con el dataset de placas bolivianas
model.train(
    #datasets
    data='C:/Users/GlobalNet/Desktop/placas_YOLOv8/datasets/Placas_Bolivia/data.yaml',
    #ciclos de entrenamiento o epocas
    epochs=50,
    #tamaño de imagen
    imgsz=640,
    #numero de lote
    batch=16,
    #nombre de salida
    name='modelo_entrenado'
)
