
import yaml
from ultralytics import YOLO  # adjust this import if using a different YOLO wrapper

# Load config from YAML
with open('train_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = YOLO('yolov8n.pt')  # or path to your custom model

# Use config values to train
# model.train(
#     data=config['data'],
#     epochs=config['epochs'],
#     imgsz=config['imgsz'],
#     batch=config['batch'],
#     patience=config['patience'],
#     name=config['name'],
#     device=config['device']
# )

results = model.train(**config)