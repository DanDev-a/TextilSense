from ultralytics import YOLO

# Cargar el modelo entrenado (debería ser un archivo .pt, no .png)
modelo_entrenado = YOLO("C:/Users/GlobalNet/Desktop/e/TextilSense/runs/detect/train/weights/best.pt/")

# Hacer predicción sobre una imagen
resultados = modelo_entrenado.predict(source="C:/Users/GlobalNet/Desktop/e/TextilSense/images.jpeg", save=True)

# Mostrar resultados
for resultado in resultados:
    print(f"\nPredicciones para la imagen: {resultado.path}")
    for caja, conf, clase in zip(resultado.boxes.xyxy, resultado.boxes.conf, resultado.boxes.cls):
        print(f"Bounding Box: {caja.tolist()} - Confianza: {conf:.2f} - Clase: {int(clase)}")
