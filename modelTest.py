from ultralytics import YOLO
import cv2
import os



# model = YOLO("models/best.pt")

# results = model("assets/pintola.png")

# print(results[0].names[results[0].probs.top1])
# print(results[0].probs.top1conf)


model_onnx = YOLO("models/best.pt")
results_onnx = model_onnx("assets/pintola.png")
print(results_onnx[0].names[results_onnx[0].probs.top1])
print(results_onnx[0].probs.top1conf)
