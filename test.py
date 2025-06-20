from ultralytics import YOLO
import torch

print(torch.cuda.is_available())

model = YOLO('yolov8n-pose.pt')

results = model('./test.jpeg')

print(type(results))
results[0].show()
