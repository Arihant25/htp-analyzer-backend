"""
YOLOv11 Training Script for HTP (House-Tree-Person) Feature Detection
Optimized for psychological assessment of house drawings.
"""

import os
import yaml
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict
import json


class HTPYOLOTrainer:
    """YOLOv11 trainer specialized for HTP house feature detection."""

    def __init__(self, data_config: str, model_size: str = "yolo11s.pt"):
        """
        Initialize trainer.

        Args:
            data_config: Path to data.yaml file
            model_size: YOLOv11 model size (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
        """
        self.data_config = data_config
        self.model_size = model_size
        self.model = None
        self.results = None

        # Create output directories
        self.output_dir = Path("results/training/training_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training configuration optimized for HTP drawings
        self.train_config = {
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "lr0": 0.01,
            "weight_decay": 0.0005,
            "mosaic": 0.8,
            "mixup": 0.1,
            "copy_paste": 0.1,
            "patience": 20,
            "save_period": 10,
            "workers": 4,
            "project": str(self.output_dir),
            "name": f'htp_yolo11s_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "SGD",
            "verbose": True,
            "seed": 42,
            "deterministic": True,
            "single_cls": False,
            "rect": False,
            "cos_lr": True,
            "close_mosaic": 10,
            "resume": False,
            "amp": True,
            "fraction": 1.0,
            "profile": False,
            "overlap_mask": True,
            "mask_ratio": 4,
            "dropout": 0.0,
            "val": True,
            "plots": True,
            "save": True,
            "save_json": True,
            "save_hybrid": False,
            "conf": 0.001,
            "iou": 0.6,
            "max_det": 300,
            "half": False,
            "dnn": False,
            "augment": False,
        }

    def setup_model(self) -> YOLO:
        """Initialize YOLOv11 model for training."""
        print(f"Setting up YOLOv11 model: {self.model_size}")

        # Initialize model
        self.model = YOLO(self.model_size)

        # Load data config to get class information
        with open(self.data_config, "r") as f:
            data_info = yaml.safe_load(f)

        print(f"Dataset classes: {data_info['names']}")
        print(f"Number of classes: {data_info['nc']}")

        return self.model

    def train(self, **kwargs) -> Dict:
        """
        Train the YOLOv11 model.

        Args:
            **kwargs: Additional training parameters to override defaults
        """
        if self.model is None:
            self.setup_model()

        # Update config with any provided kwargs
        config = self.train_config.copy()
        config.update(kwargs)
        config["data"] = self.data_config

        print("Starting training with configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Start training
        print("\n" + "=" * 50)
        print("STARTING HTP YOLO TRAINING")
        print("=" * 50)

        self.results = self.model.train(**config)

        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)

        # Save training configuration
        config_save_path = self.output_dir / config["name"] / "train_config.yaml"
        config_save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return self.results

    def validate(self, data_split: str = "val") -> Dict:
        """Validate the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print(f"Validating model on {data_split} split...")
        val_results = self.model.val(split=data_split)

        return val_results

    def analyze_training_results(self) -> Dict:
        """Analyze and visualize training results."""
        if self.results is None:
            raise ValueError("No training results available. Train the model first.")

        # Get the results directory
        results_dir = Path(self.results.save_dir)

        analysis = {
            "model_path": str(results_dir / "weights" / "best.pt"),
            "results_dir": str(results_dir),
            "metrics": {},
            "plots": [],
        }

        # Read results CSV if available
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv)

            # Get final metrics
            final_metrics = df.iloc[-1].to_dict()
            analysis["metrics"] = {
                "final_epoch": len(df),
                "best_mAP50": df["metrics/mAP50(B)"].max(),
                "best_mAP50_95": df["metrics/mAP50-95(B)"].max(),
                "final_train_loss": final_metrics.get("train/box_loss", 0),
                "final_val_loss": final_metrics.get("val/box_loss", 0),
            }

            # Create training plots
            self._create_training_plots(df, results_dir)
            analysis["plots"].append(str(results_dir / "training_curves.png"))

        # Save analysis
        analysis_file = results_dir / "training_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"Training analysis saved to: {analysis_file}")
        return analysis

    def _create_training_plots(self, df: pd.DataFrame, save_dir: Path):
        """Create training visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("HTP YOLOv11 Training Results", fontsize=16)

        # Loss curves
        axes[0, 0].plot(
            df.index, df["train/box_loss"], label="Train Box Loss", color="blue"
        )
        axes[0, 0].plot(
            df.index, df["val/box_loss"], label="Val Box Loss", color="orange"
        )
        axes[0, 0].set_title("Box Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # mAP curves
        axes[0, 1].plot(
            df.index, df["metrics/mAP50(B)"], label="mAP@0.5", color="green"
        )
        axes[0, 1].plot(
            df.index, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="red"
        )
        axes[0, 1].set_title("Mean Average Precision")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("mAP")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision and Recall
        axes[1, 0].plot(
            df.index, df["metrics/precision(B)"], label="Precision", color="purple"
        )
        axes[1, 0].plot(
            df.index, df["metrics/recall(B)"], label="Recall", color="brown"
        )
        axes[1, 0].set_title("Precision and Recall")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate
        if "lr/pg0" in df.columns:
            axes[1, 1].plot(
                df.index, df["lr/pg0"], label="Learning Rate", color="black"
            )
            axes[1, 1].set_title("Learning Rate Schedule")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

    def export_model(self, format: str = "onnx", **kwargs) -> str:
        """Export trained model to different formats."""
        if self.model is None:
            raise ValueError("No model available for export. Train the model first.")

        print(f"Exporting model to {format} format...")

        # Get the best model
        best_model_path = Path(self.results.save_dir) / "weights" / "best.pt"
        model = YOLO(best_model_path)

        # Export
        export_path = model.export(format=format, **kwargs)

        print(f"Model exported to: {export_path}")
        return export_path

    def create_training_report(self) -> str:
        """Create a comprehensive training report."""
        if self.results is None:
            raise ValueError("No training results available.")

        results_dir = Path(self.results.save_dir)
        report_path = results_dir / "training_report.md"

        # Load analysis
        analysis_file = results_dir / "training_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, "r") as f:
                analysis = json.load(f)
        else:
            analysis = self.analyze_training_results()

        # Create report
        report = f"""# HTP YOLOv11 Training Report

## Training Configuration
- Model: {self.model_size}
- Data Config: {self.data_config}
- Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Results Summary
- **Final Epoch**: {analysis['metrics'].get('final_epoch', 'N/A')}
- **Best mAP@0.5**: {analysis['metrics'].get('best_mAP50', 'N/A'):.4f}
- **Best mAP@0.5:0.95**: {analysis['metrics'].get('best_mAP50_95', 'N/A'):.4f}
- **Final Training Loss**: {analysis['metrics'].get('final_train_loss', 'N/A'):.4f}
- **Final Validation Loss**: {analysis['metrics'].get('final_val_loss', 'N/A'):.4f}

## Model Files
- **Best Model**: `{analysis.get('model_path', 'N/A')}`
- **Results Directory**: `{analysis.get('results_dir', 'N/A')}`

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
model = YOLO('{analysis.get('model_path', 'best.pt')}')

# Run inference
results = model('path/to/house_drawing.jpg')

# Process results for HTP analysis
# (See feature_extractor.py for detailed analysis)
```
"""

        with open(report_path, "w") as f:
            f.write(report)

        print(f"Training report saved to: {report_path}")
        return str(report_path)


def main():
    """Main training function."""
    # Configuration
    data_config = "house_only_dataset/data.yaml"  # Assuming filtered dataset exists
    model_size = "yolo11s.pt"  # Small model for faster training

    # Check if data config exists
    if not os.path.exists(data_config):
        print(f"Data config not found: {data_config}")
        print("Please run data_processor.py first to create the filtered dataset.")
        return

    # Initialize trainer
    trainer = HTPYOLOTrainer(data_config=data_config, model_size=model_size)

    # Custom training parameters for HTP dataset
    custom_params = {
        "epochs": 150,  # More epochs for better convergence
        "batch": 8,  # Smaller batch size for limited data
        "imgsz": 640,  # Standard image size
        "patience": 30,  # More patience for small dataset
    }

    # Train the model
    print("Starting HTP YOLOv11 training...")
    results = trainer.train(**custom_params)

    # Validate the model
    print("\nValidating trained model...")
    val_results = trainer.validate()

    # Analyze results
    print("\nAnalyzing training results...")
    analysis = trainer.analyze_training_results()

    # Create report
    print("\nCreating training report...")
    report_path = trainer.create_training_report()

    # Export model (optional)
    print("\nExporting model to ONNX format...")
    try:
        export_path = trainer.export_model(format="onnx")
        print(f"Model exported to: {export_path}")
    except Exception as e:
        print(f"Export failed: {e}")

    print("\n" + "=" * 50)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 50)
    print(f"Best model: {analysis.get('model_path', 'N/A')}")
    print(f"Training report: {report_path}")


if __name__ == "__main__":
    main()
