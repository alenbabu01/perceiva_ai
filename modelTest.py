import os
from ultralytics import YOLO

# Load the trained YOLO classification model
model = YOLO('models/yolo-cls.pt')

# Create output directory if it doesn't exist
output_dir = "runs/custom_predict"
os.makedirs(output_dir, exist_ok=True)

# Run prediction
results = model.predict('assets/maggi.png', conf=0.5)

# Save annotated prediction image(s)
for i, result in enumerate(results):
    output_path = os.path.join(output_dir, f"cola_pred_{i}.jpg")
    result.save(filename=output_path)
    print("Saved annotated image to:", output_path)

print(results)
