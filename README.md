# TextilSense
bash :
pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install labelme
labelme 
pip install ultralytics
yolo task=segment mode=train epochs=60 data=dataset.yaml model=yolov8x-seg.pt imgsz=640 batch=4 name='prueba'

val es recomendable que tenga imagenes diferentes a train
ejemplo: train 1000 imagenes - val 200 imagenes