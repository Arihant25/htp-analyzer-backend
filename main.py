"""
Main execution script for HTP (House-Tree-Person) YOLOv11 training pipeline.
Complete workflow from data processing to model evaluation and feature analysis.
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processor import HTPDataProcessor
from train_yolo import HTPYOLOTrainer
from feature_extractor import HTPFeatureExtractor
from model_evaluator import HTPModelEvaluator


def main():
    """Main pipeline execution with command line arguments."""
    parser = argparse.ArgumentParser(
        description="HTP YOLOv11 Training and Analysis Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "train", "evaluate", "analyze", "full"],
        default="full",
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--model-size", type=str, default="yolo11s.pt", help="YOLOv11 model size"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for inference",
    )
    parser.add_argument(
        "--image-path", type=str, help="Path to single image for analysis"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory for results"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HTP (House-Tree-Person) YOLOv11 Analysis Pipeline")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data Root: {args.data_root}")
    print("=" * 60)

    try:
        if args.mode == "preprocess" or args.mode == "full":
            run_preprocessing(args)

        if args.mode == "train" or args.mode == "full":
            run_training(args)

        if args.mode == "evaluate" or args.mode == "full":
            run_evaluation(args)

        if args.mode == "analyze":
            run_analysis(args)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Pipeline execution failed.")
        sys.exit(1)


def run_preprocessing(args):
    """Run data preprocessing step."""
    print("\n🔄 STEP 1: DATA PREPROCESSING")
    print("-" * 40)

    # Initialize processor
    processor = HTPDataProcessor(data_root=args.data_root)

    # Filter dataset for house-only classes
    print("Creating house-only dataset...")
    house_dataset_path, stats = processor.filter_house_only_dataset()
    print(f"✅ House-only dataset created at: {house_dataset_path}")
    print(f"📊 Dataset statistics: {stats}")

    # Analyze dataset
    print("\nAnalyzing dataset statistics...")
    dataset_stats = processor.analyze_dataset_statistics(house_dataset_path)

    print("\nDataset Summary:")
    for split, split_stats in dataset_stats["splits"].items():
        if split_stats["images"] > 0:
            print(
                f"  {split}: {split_stats['images']} images, {split_stats['annotations']} annotations"
            )

    print("\nClass Distribution:")
    for class_name, count in dataset_stats["class_distribution"].items():
        print(f"  {class_name}: {count}")


def run_training(args):
    """Run model training step."""
    print("\n🚀 STEP 2: MODEL TRAINING")
    print("-" * 40)

    # Check if preprocessed data exists
    data_config = "data/processed/house_only_dataset/data.yaml"
    if not Path(data_config).exists():
        print("❌ Preprocessed dataset not found. Running preprocessing first...")
        run_preprocessing(args)

    # Initialize trainer
    trainer = HTPYOLOTrainer(data_config=data_config, model_size=args.model_size)

    # Custom training parameters
    train_params = {
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": 640,
        "patience": 30,
        "save_period": 10,
    }

    print(f"Training with parameters: {train_params}")

    # Train the model
    results = trainer.train(**train_params)
    print(f"✅ Training completed. Results saved to: {results.save_dir}")

    # Analyze results
    print("Analyzing training results...")
    analysis = trainer.analyze_training_results()

    # Create report
    report_path = trainer.create_training_report()
    print(f"✅ Training report created: {report_path}")

    # Export model
    try:
        export_path = trainer.export_model(format="onnx")
        print(f"✅ Model exported to ONNX: {export_path}")
    except Exception as e:
        print(f"⚠️ Export failed: {e}")


def run_evaluation(args):
    """Run model evaluation step."""
    print("\n📊 STEP 3: MODEL EVALUATION")
    print("-" * 40)

    # Find the latest trained model
    model_path = find_latest_model()
    if not model_path:
        print("❌ No trained model found. Please run training first.")
        return

    data_config = "data/processed/house_only_dataset/data.yaml"
    if not Path(data_config).exists():
        print("❌ Data config not found. Please run preprocessing first.")
        return

    print(f"Evaluating model: {model_path}")

    # Initialize evaluator
    evaluator = HTPModelEvaluator(model_path, data_config)

    # Run evaluation
    results = evaluator.evaluate_model(confidence_threshold=args.confidence)

    # Create visualizations
    viz_paths = evaluator.visualize_results()

    print("✅ Evaluation completed!")
    print(f"📊 Results directory: {evaluator.output_dir}")
    print(f"📈 Visualizations created: {len(viz_paths)}")

    # Print key metrics
    if "yolo_metrics" in results:
        metrics = results["yolo_metrics"]
        print(f"\nKey Metrics:")
        print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

    if "htp_specific_metrics" in results:
        htp_score = results["htp_specific_metrics"]["overall_htp_score"]
        print(f"  HTP Score: {htp_score:.4f}")


def run_analysis(args):
    """Run feature analysis on a single image or batch."""
    print("\n🔍 STEP 4: FEATURE ANALYSIS")
    print("-" * 40)

    # Find the latest trained model
    model_path = find_latest_model()
    if not model_path:
        print("❌ No trained model found. Please run training first.")
        return

    print(f"Using model: {model_path}")

    # Initialize extractor
    extractor = HTPFeatureExtractor(model_path)

    if args.image_path:
        # Single image analysis
        if not Path(args.image_path).exists():
            print(f"❌ Image not found: {args.image_path}")
            return

        print(f"Analyzing image: {args.image_path}")
        analysis = extractor.analyze_image(
            args.image_path, confidence_threshold=args.confidence
        )

        # Generate report and visualization
        report_path = extractor.generate_report(analysis)
        viz_path = extractor.visualize_analysis(analysis)

        print(f"✅ Analysis completed!")
        print(f"📄 Report: {report_path}")
        print(f"📊 Visualization: {viz_path}")

        # Print summary
        print(f"\nAnalysis Summary:")
        print(f"  House size: {analysis.house_size_category}")
        print(f"  Detected features: {', '.join(analysis.detected_features)}")
        print(f"  Missing features: {', '.join(analysis.missing_features)}")
        print(f"  Risk factors: {len(analysis.risk_factors)}")
        print(f"  Positive indicators: {len(analysis.positive_indicators)}")

    else:
        # Batch analysis on test set
        test_dir = "data/raw/test/images"
        if Path(test_dir).exists():
            print(f"Running batch analysis on: {test_dir}")
            batch_df = extractor.batch_analysis(test_dir, args.output_dir)
            print(f"✅ Batch analysis completed. Processed {len(batch_df)} images.")
            print(f"📊 Results saved to: {args.output_dir}")
        else:
            print(f"❌ Test directory not found: {test_dir}")


def find_latest_model():
    """Find the latest trained model."""
    training_outputs = Path("results/training/training_outputs")
    if not training_outputs.exists():
        return None

    # Look for the latest training run
    runs = list(training_outputs.glob("htp_yolo11s_*"))
    if not runs:
        return None

    # Get the most recent run
    latest_run = max(runs, key=lambda x: x.stat().st_mtime)
    model_path = latest_run / "weights" / "best.pt"

    return str(model_path) if model_path.exists() else None


def print_usage_examples():
    """Print usage examples."""
    print(
        """
Usage Examples:

1. Full pipeline (preprocess + train + evaluate):
   python main.py --mode full --epochs 150 --batch-size 8

2. Preprocess data only:
   python main.py --mode preprocess

3. Train model only:
   python main.py --mode train --epochs 100 --batch-size 16

4. Evaluate trained model:
   python main.py --mode evaluate --confidence 0.3

5. Analyze single image:
   python main.py --mode analyze --image-path "path/to/image.jpg"

6. Batch analysis on test set:
   python main.py --mode analyze --output-dir "analysis_results"

Options:
  --model-size: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
  --epochs: Number of training epochs (default: 100)
  --batch-size: Training batch size (default: 16)
  --confidence: Detection confidence threshold (default: 0.25)
"""
    )


if __name__ == "__main__":
    # Check if help is requested
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_usage_examples()
    else:
        main()
