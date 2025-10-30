# HTP YOLOv11 Training Report

## Training Configuration
- Model: yolo11s.pt
- Data Config: data/processed/house_only_dataset/data.yaml
- Training Date: 2025-10-30 06:12:45

## Results Summary
- **Final Epoch**: 72
- **Best mAP@0.5**: 0.8483
- **Best mAP@0.5:0.95**: 0.6857
- **Final Training Loss**: 0.6419
- **Final Validation Loss**: 0.8225

## Model Files
- **Best Model**: `C:\Users\Arihant\Git\htp-analyzer-backend\results\training\training_outputs\htp_yolo11s_20251029_230000\weights\best.pt`
- **Results Directory**: `C:\Users\Arihant\Git\htp-analyzer-backend\results\training\training_outputs\htp_yolo11s_20251029_230000`

## Training Plots
- Training curves saved as: `training_curves.png`

## HTP Feature Detection Focus
This model was specifically trained to detect house features relevant for psychological assessment:
- **Chimney**: Indicator of warmth in home environment
- **Door**: Social accessibility and interpersonal connection
- **House**: Overall psychological state representation  
- **Roof**: Protection and security feelings
- **Wall**: Ego boundaries and self-protection
- **Window**: Openness to environment and social interaction

## Usage
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('C:\Users\Arihant\Git\htp-analyzer-backend\results\training\training_outputs\htp_yolo11s_20251029_230000\weights\best.pt')

# Run inference
results = model('path/to/house_drawing.jpg')

# Process results for HTP analysis
# (See feature_extractor.py for detailed analysis)
```
