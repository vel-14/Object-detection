from ultralytics import YOLO

# load a pretrained model 
model = YOLO("yolov8n.pt")  

# train the model for our dataset
model.train(data="/Users/velmurugan/Desktop/@/python_works/computer vision/config.yaml", epochs=100)  