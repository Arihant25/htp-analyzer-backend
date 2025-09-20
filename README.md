# HTP Analyzer Backend - House-Tree-Person Psychological Assessment

A comprehensive YOLOv11-based backend system for automated detection and analysis of house features in House-Tree-Person (HTP) psychological assessment drawings.

## 🎯 Overview

This project implements a complete pipeline for training YOLOv11 models to detect and analyze house features in psychological drawings according to HTP assessment guidelines. The system can identify key psychological indicators such as:

- **House size** (withdrawal vs aggression indicators)
- **Door characteristics** (social accessibility, fearfulness)
- **Window presence** (openness vs defensiveness) 
- **Chimney features** (emotional warmth indicators)
- **Structural elements** (walls, roof - security indicators)
- **Overall placement** (personality orientation indicators)

## 🚀 Features

- **Data Processing**: Filters dataset for house-specific features
- **YOLOv11 Training**: Optimized training pipeline for psychological drawings
- **Feature Analysis**: Automated HTP psychological assessment
- **Comprehensive Evaluation**: Model performance metrics and visualizations
- **Batch Processing**: Analyze multiple drawings simultaneously
- **Interactive Reports**: Detailed psychological interpretation reports

## 📋 Dataset Classes

The system detects 6 key house features:
- `chimney` - Emotional warmth indicators
- `door` - Social accessibility and connection
- `house` - Overall psychological state
- `roof` - Protection and security feelings
- `wall` - Ego boundaries and self-protection  
- `window` - Environmental openness and interaction

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd htp-analyzer-backend
```

2. **Install dependencies**:
```bash
uv sync
```

## 🔧 Usage

### Command Line Interface

The main script provides a comprehensive pipeline with multiple modes:

```bash
# Full pipeline (preprocess + train + evaluate)
python main.py --mode full --epochs 150 --batch-size 8

# Individual steps
python main.py --mode preprocess          # Data preprocessing only
python main.py --mode train --epochs 100  # Training only  
python main.py --mode evaluate            # Evaluation only
python main.py --mode analyze --image-path "data/raw/test/images/sample.jpg"  # Single image analysis

# Help
python main.py --help
```

### Python API

```python
from src.data_processor import HTPDataProcessor
from src.train_yolo import HTPYOLOTrainer
from src.feature_extractor import HTPFeatureExtractor

# 1. Process data
processor = HTPDataProcessor("data/raw")
house_dataset_path, stats = processor.filter_house_only_dataset()

# 2. Train model
trainer = HTPYOLOTrainer("data/processed/house_only_dataset/data.yaml")
results = trainer.train(epochs=100, batch=16)

# 3. Analyze images
extractor = HTPFeatureExtractor("results/training/training_outputs/latest/weights/best.pt")
analysis = extractor.analyze_image("drawing.jpg")
report = extractor.generate_report(analysis)
```

## 📊 Pipeline Components

### 1. Data Processor (`src/data_processor.py`)
- Filters dataset for house-related classes only
- Analyzes dataset statistics and class distributions
- Creates feature analysis datasets
- Performs HTP-specific data preparation

### 2. YOLO Trainer (`src/train_yolo.py`) 
- YOLOv11 model initialization and configuration
- Optimized hyperparameters for psychological drawings
- Training monitoring and result analysis
- Model export capabilities (ONNX, etc.)

### 3. Feature Extractor (`src/feature_extractor.py`)
- Automated HTP psychological assessment
- Detailed feature analysis according to HTP guidelines
- Risk factor and positive indicator identification
- Individual and batch processing capabilities

### 4. Model Evaluator (`src/model_evaluator.py`)
- Comprehensive model performance evaluation
- HTP-specific metrics (psychological completeness)
- Interactive visualizations and dashboards
- Detailed evaluation reports

## 🧠 Psychological Interpretation

The system interprets detected features according to established HTP guidelines:

### House Size Analysis
- **Small** (<10% of image): Withdrawal, inadequacy, rejection of home life
- **Large** (>60% of image): Frustration, hostility, aggressive tendencies
- **Normal**: Balanced self-perception

### Feature Presence/Absence
- **Missing Door**: Insecurity, difficulty connecting with others
- **Missing Windows**: Guarded, suspicious, hostile tendencies
- **Missing Chimney**: Lack of warmth in home environment

### Placement Analysis
- **Left Side**: Past-focused, impulsive, emotional orientation
- **Right Side**: Future-focused, intellectual, reality-oriented
- **High Placement**: Optimism, ambition, active fantasy life
- **Low Placement**: Insecurity, low self-esteem, depression tendencies

## 📈 Output Examples

### Analysis Report
```markdown
# HTP Analysis Report

## Detection Summary
- Detected Features: house, door, window, roof
- Missing Features: chimney, wall

## Psychological Analysis
- House Size: normal (0.234 area ratio)
- Placement: center and middle (balanced)
- Risk Factors: lack of emotional warmth
- Positive Indicators: appropriate size perception, social accessibility
```

### Metrics Dashboard
- Model performance metrics (mAP, precision, recall)
- Class-wise detection rates
- HTP psychological completeness scores
- Interactive visualization dashboards

## 🎯 Key Metrics

### Model Performance
- **mAP@0.5**: Mean Average Precision at 0.5 IoU
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision/Recall**: Per-class detection accuracy

### HTP-Specific Metrics  
- **Psychological Completeness**: Percentage of essential features detected
- **Essential Feature Detection Rate**: Success rate for critical features
- **Overall HTP Score**: Composite assessment suitability score

## 📁 Project Structure

```
htp-analyzer-backend/
├── main.py                          # Main execution script
├── src/                             # Source code modules
│   ├── data_processor.py            # Dataset preprocessing
│   ├── train_yolo.py               # Model training
│   ├── feature_extractor.py        # Feature analysis
│   └── model_evaluator.py          # Model evaluation
├── data/                            # All datasets
│   ├── raw/                         # Original datasets
│   │   ├── train/                   # Raw training data
│   │   ├── valid/                   # Raw validation data
│   │   └── test/                    # Raw test data
│   └── processed/                   # Processed datasets
│       └── house_only_dataset/      # Filtered house-only dataset
├── results/                         # All training and evaluation outputs
│   ├── training/                    # Training outputs
│   │   └── training_outputs/        # Model weights and training logs
│   ├── evaluation/                  # Evaluation results
│   │   └── evaluation_results/      # Metrics and visualizations
│   └── validation/                  # Validation outputs
│       └── runs/                    # YOLO validation runs
├── data.yaml                        # Main dataset configuration
├── yolo11s.pt                      # Pre-trained model weights
└── HTP Checklist.txt               # Psychological assessment guidelines
```

### Directory Organization

- **data/raw/**: Original, unprocessed datasets as downloaded
- **data/processed/**: Datasets that have been filtered or processed
- **results/training/**: All training outputs, model weights, and logs
- **results/evaluation/**: Model evaluation reports and visualizations
- **results/validation/**: YOLO validation runs and intermediate results

This organized structure provides:
- **Clear separation**: Data, code, and results are distinctly organized
- **Scalability**: Easy to add new datasets or result types
- **Maintainability**: Simplified backup and version control
- **Traceability**: Clear lineage from raw data to final results

## 🔬 Research Applications

This system supports research in:
- **Automated Psychological Assessment**: Reducing manual analysis time
- **Digital Mental Health**: Scalable screening tools
- **Art Therapy Analysis**: Objective interpretation of therapeutic drawings

## ⚠️ Important Notes

### Clinical Use Disclaimer
- This system is designed to **support** psychological assessment, not replace professional judgment
- All automated analyses should be **validated by qualified professionals**
- The system may not detect subtle artistic variations or cultural differences
- Use higher confidence thresholds for critical assessments

### Limitations
- Performance may vary with different drawing styles and quality
- Missing features could indicate either true absence or detection failure
- Psychological interpretation requires human expertise and context
- Model trained on specific dataset may not generalize to all populations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📞 Support

For questions, issues, or contributions:
- Create an issue in the GitHub repository
- Check the documentation and examples
- Review the HTP Checklist.txt for psychological interpretation guidelines

---

**Note**: This tool is intended for research and educational purposes. For clinical applications, always consult with qualified mental health professionals.
