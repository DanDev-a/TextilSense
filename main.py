#val se tienen imagenes diferentes a train
#ejemplo: train 1000 imagenes - val 200 imagenes# Importamos las librerias
from ultralytics import YOLO
import cv2
import time
import threading

# Cargar modelo
model = YOLO("best.pt")
# model.to("cuda")  # Descomenta si tienes GPU (muy recomendable)

# Iniciar cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Variable global para almacenar el último frame procesado
last_result = None
lock = threading.Lock()

# Función para inferencia en segundo plano
def detectar_en_segundo_plano():
    global last_result
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = model.predict(frame, imgsz=416, conf=0.5, verbose=False)[0]
        with lock:
            last_result = result

# Iniciar hilo de inferencia
thread = threading.Thread(target=detectar_en_segundo_plano, daemon=True)
thread.start()

# Bucle principal
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar el último resultado procesado
    with lock:
        result = last_result

    if result is not None:
        annotated = result.plot()
    else:
        annotated = frame

    # Mostrar FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(annotated, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar imagen
    cv2.imshow("DETECCIÓN EN VIVO", annotated)

    # Salida con tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
