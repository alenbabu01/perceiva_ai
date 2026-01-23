from ultralytics import YOLO
from PIL import Image
import cv2

# Load the YOLOv8 model
model = YOLO(r"models\detectionModel\best (1).pt")

# Run inference on the test image
test_image_path = "assets\shelf2.jpeg"  # Replace with your test image path
results = model.predict(source=test_image_path, conf=0.25)

# Load the image
image = cv2.imread(test_image_path)

# Annotate detections on the image
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put label with class name and confidence
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the annotated image
output_path = "detected_objects.jpg"
cv2.imwrite(output_path, image)
print(f"Annotated image saved to {output_path}")