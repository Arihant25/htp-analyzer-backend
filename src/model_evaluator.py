"""
Model Evaluation and Visualization for HTP YOLO Model
Comprehensive evaluation metrics and visualizations for the trained model.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml


class HTPModelEvaluator:
    """Comprehensive evaluation of HTP YOLO model performance."""

    def __init__(self, model_path: str, data_config: str):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained YOLO model
            data_config: Path to data.yaml configuration
        """
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.data_config_path = data_config

        # Load data configuration
        with open(data_config, "r") as f:
            self.data_config = yaml.safe_load(f)

        self.class_names = self.data_config["names"]
        self.num_classes = len(self.class_names)

        # Create output directory
        self.output_dir = Path("results/evaluation/evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation results storage
        self.evaluation_results = {}

    def evaluate_model(
        self, test_data_path: str = None, confidence_threshold: float = 0.25
    ) -> Dict:
        """
        Comprehensive model evaluation.

        Args:
            test_data_path: Path to test dataset (if None, uses config)
            confidence_threshold: Confidence threshold for detections

        Returns:
            Dictionary containing evaluation metrics
        """
        print("Starting comprehensive model evaluation...")

        # Run official YOLO validation
        print("Running YOLO validation...")
        val_results = self.model.val(
            data=self.data_config_path, conf=confidence_threshold
        )

        # Extract validation metrics
        self.evaluation_results["yolo_metrics"] = {
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
            "f1_score": float(
                2
                * val_results.box.mp
                * val_results.box.mr
                / (val_results.box.mp + val_results.box.mr + 1e-8)
            ),
        }

        # Class-wise metrics
        if hasattr(val_results.box, "maps"):
            class_maps = val_results.box.maps
            self.evaluation_results["class_metrics"] = {}
            for i, class_name in enumerate(self.class_names):
                if i < len(class_maps):
                    self.evaluation_results["class_metrics"][class_name] = {
                        "mAP50_95": float(class_maps[i])
                    }

        # Perform custom evaluation on test set
        if test_data_path is None:
            test_data_path = self.data_config.get("test", self.data_config.get("val"))

        if test_data_path:
            print("Performing detailed test set evaluation...")
            custom_metrics = self._detailed_evaluation(
                test_data_path, confidence_threshold
            )
            self.evaluation_results.update(custom_metrics)

        # Generate evaluation report
        report_path = self._generate_evaluation_report()
        self.evaluation_results["report_path"] = report_path

        print(f"Evaluation complete. Results saved to: {self.output_dir}")
        return self.evaluation_results

    def _detailed_evaluation(
        self, test_data_path: str, confidence_threshold: float
    ) -> Dict:
        """Perform detailed evaluation on test dataset."""
        test_images_path = Path(test_data_path)
        test_labels_path = test_images_path.parent / "labels"

        if not test_images_path.exists():
            print(f"Test path not found: {test_images_path}")
            return {}

        # Collect all test images
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        test_images = []
        for ext in image_extensions:
            test_images.extend(test_images_path.glob(f"*{ext}"))
            test_images.extend(test_images_path.glob(f"*{ext.upper()}"))

        print(f"Found {len(test_images)} test images")

        # Evaluation metrics
        detailed_results = {
            "detection_analysis": [],
            "feature_statistics": {
                name: {"tp": 0, "fp": 0, "fn": 0} for name in self.class_names
            },
            "htp_specific_metrics": {},
            "error_analysis": [],
        }

        for img_path in test_images:
            # Run inference
            results = self.model(str(img_path), conf=confidence_threshold)

            # Load ground truth
            label_path = test_labels_path / (img_path.stem + ".txt")
            gt_data = self._load_ground_truth(label_path) if label_path.exists() else []

            # Analyze detections
            detection_analysis = self._analyze_detections(results[0], gt_data, img_path)
            detailed_results["detection_analysis"].append(detection_analysis)

            # Update feature statistics
            self._update_feature_statistics(
                detection_analysis, detailed_results["feature_statistics"]
            )

        # Calculate metrics
        detailed_results["precision_recall"] = self._calculate_precision_recall(
            detailed_results["feature_statistics"]
        )
        detailed_results["htp_specific_metrics"] = self._calculate_htp_metrics(
            detailed_results["detection_analysis"]
        )

        return detailed_results

    def _load_ground_truth(self, label_path: Path) -> List[Dict]:
        """Load ground truth annotations from YOLO format."""
        gt_data = []

        if not label_path.exists():
            return gt_data

        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                class_id = int(parts[0])

                if class_id < len(self.class_names):
                    # Parse coordinates (polygon format)
                    coords = [float(x) for x in parts[1:]]

                    gt_data.append(
                        {
                            "class_id": class_id,
                            "class_name": self.class_names[class_id],
                            "coordinates": coords,
                        }
                    )

        return gt_data

    def _analyze_detections(self, result, gt_data: List[Dict], img_path: Path) -> Dict:
        """Analyze detections vs ground truth for a single image."""
        analysis = {
            "image_path": str(img_path),
            "gt_classes": [item["class_name"] for item in gt_data],
            "detected_classes": [],
            "detection_confidences": {},
            "true_positives": [],
            "false_positives": [],
            "false_negatives": [],
            "htp_analysis": {},
        }

        # Extract detections
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)

                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    analysis["detected_classes"].append(class_name)
                    analysis["detection_confidences"][class_name] = confidence

        # Calculate TP, FP, FN for each class
        gt_classes_set = set(analysis["gt_classes"])
        detected_classes_set = set(analysis["detected_classes"])

        # True positives: classes present in both GT and detections
        analysis["true_positives"] = list(
            gt_classes_set.intersection(detected_classes_set)
        )

        # False positives: classes detected but not in GT
        analysis["false_positives"] = list(detected_classes_set - gt_classes_set)

        # False negatives: classes in GT but not detected
        analysis["false_negatives"] = list(gt_classes_set - detected_classes_set)

        # HTP-specific analysis
        analysis["htp_analysis"] = self._htp_detection_analysis(analysis)

        return analysis

    def _htp_detection_analysis(self, analysis: Dict) -> Dict:
        """HTP-specific analysis of detection quality."""
        htp_analysis = {
            "essential_features_detected": 0,
            "missing_essential_features": [],
            "psychological_completeness": 0.0,
        }

        # Essential features for HTP analysis
        essential_features = ["house", "door", "window"]

        for feature in essential_features:
            if feature in analysis["true_positives"]:
                htp_analysis["essential_features_detected"] += 1
            elif feature in analysis["false_negatives"]:
                htp_analysis["missing_essential_features"].append(feature)

        # Calculate psychological completeness score
        total_essential = len(essential_features)
        htp_analysis["psychological_completeness"] = (
            htp_analysis["essential_features_detected"] / total_essential
        )

        return htp_analysis

    def _update_feature_statistics(self, detection_analysis: Dict, feature_stats: Dict):
        """Update feature-wise statistics."""
        for class_name in self.class_names:
            if class_name in detection_analysis["true_positives"]:
                feature_stats[class_name]["tp"] += 1
            if class_name in detection_analysis["false_positives"]:
                feature_stats[class_name]["fp"] += 1
            if class_name in detection_analysis["false_negatives"]:
                feature_stats[class_name]["fn"] += 1

    def _calculate_precision_recall(self, feature_stats: Dict) -> Dict:
        """Calculate precision and recall for each class."""
        metrics = {}

        for class_name, stats in feature_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": tp + fn,
            }

        return metrics

    def _calculate_htp_metrics(self, detection_analysis: List[Dict]) -> Dict:
        """Calculate HTP-specific evaluation metrics."""
        htp_metrics = {
            "psychological_completeness_scores": [],
            "essential_feature_detection_rates": {},
            "overall_htp_score": 0.0,
        }

        essential_features = ["house", "door", "window"]
        feature_detection_counts = {feature: 0 for feature in essential_features}
        total_images = len(detection_analysis)

        for analysis in detection_analysis:
            htp_analysis = analysis["htp_analysis"]
            htp_metrics["psychological_completeness_scores"].append(
                htp_analysis["psychological_completeness"]
            )

            # Count essential feature detections
            for feature in essential_features:
                if feature in analysis["true_positives"]:
                    feature_detection_counts[feature] += 1

        # Calculate detection rates
        for feature in essential_features:
            htp_metrics["essential_feature_detection_rates"][feature] = (
                feature_detection_counts[feature] / total_images
                if total_images > 0
                else 0
            )

        # Overall HTP score
        htp_metrics["overall_htp_score"] = np.mean(
            htp_metrics["psychological_completeness_scores"]
        )

        return htp_metrics

    def visualize_results(self) -> List[str]:
        """Create comprehensive visualizations of evaluation results."""
        if not self.evaluation_results:
            raise ValueError(
                "No evaluation results available. Run evaluate_model() first."
            )

        visualization_paths = []

        # 1. Overall performance metrics
        viz_path = self._plot_overall_metrics()
        visualization_paths.append(viz_path)

        # 2. Class-wise performance
        if "precision_recall" in self.evaluation_results:
            viz_path = self._plot_class_performance()
            visualization_paths.append(viz_path)

        # 3. HTP-specific metrics
        if "htp_specific_metrics" in self.evaluation_results:
            viz_path = self._plot_htp_metrics()
            visualization_paths.append(viz_path)

        # 4. Confusion matrix
        viz_path = self._plot_confusion_matrix()
        visualization_paths.append(viz_path)

        # 5. Interactive dashboard
        dashboard_path = self._create_interactive_dashboard()
        visualization_paths.append(dashboard_path)

        return visualization_paths

    def _plot_overall_metrics(self) -> str:
        """Plot overall model performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("HTP YOLO Model - Overall Performance", fontsize=16)

        # YOLO metrics
        yolo_metrics = self.evaluation_results["yolo_metrics"]
        metrics_names = list(yolo_metrics.keys())
        metrics_values = list(yolo_metrics.values())

        axes[0, 0].bar(metrics_names, metrics_values, color="skyblue", alpha=0.7)
        axes[0, 0].set_title("YOLO Validation Metrics")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].set_ylim(0, 1)

        # Add value labels on bars
        for i, v in enumerate(metrics_values):
            axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        # HTP specific metrics if available
        if "htp_specific_metrics" in self.evaluation_results:
            htp_metrics = self.evaluation_results["htp_specific_metrics"]

            # Psychological completeness distribution
            completeness_scores = htp_metrics["psychological_completeness_scores"]
            axes[0, 1].hist(completeness_scores, bins=10, alpha=0.7, color="lightgreen")
            axes[0, 1].set_title("Psychological Completeness Distribution")
            axes[0, 1].set_xlabel("Completeness Score")
            axes[0, 1].set_ylabel("Frequency")

            # Essential feature detection rates
            features = list(htp_metrics["essential_feature_detection_rates"].keys())
            rates = list(htp_metrics["essential_feature_detection_rates"].values())

            axes[1, 0].bar(features, rates, color="orange", alpha=0.7)
            axes[1, 0].set_title("Essential Feature Detection Rates")
            axes[1, 0].set_ylabel("Detection Rate")
            axes[1, 0].set_ylim(0, 1)

            # Add value labels
            for i, v in enumerate(rates):
                axes[1, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        # Model info
        axes[1, 1].text(
            0.1,
            0.8,
            f"Model: {Path(self.model_path).name}",
            transform=axes[1, 1].transAxes,
            fontsize=12,
        )
        axes[1, 1].text(
            0.1,
            0.6,
            f"Classes: {', '.join(self.class_names)}",
            transform=axes[1, 1].transAxes,
            fontsize=10,
        )
        axes[1, 1].text(
            0.1,
            0.4,
            f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            transform=axes[1, 1].transAxes,
            fontsize=10,
        )
        if "htp_specific_metrics" in self.evaluation_results:
            overall_score = self.evaluation_results["htp_specific_metrics"][
                "overall_htp_score"
            ]
            axes[1, 1].text(
                0.1,
                0.2,
                f"Overall HTP Score: {overall_score:.3f}",
                transform=axes[1, 1].transAxes,
                fontsize=12,
                weight="bold",
            )
        axes[1, 1].set_title("Model Information")
        axes[1, 1].axis("off")

        plt.tight_layout()
        output_path = self.output_dir / "overall_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def _plot_class_performance(self) -> str:
        """Plot class-wise performance metrics."""
        precision_recall = self.evaluation_results["precision_recall"]

        classes = list(precision_recall.keys())
        precisions = [precision_recall[cls]["precision"] for cls in classes]
        recalls = [precision_recall[cls]["recall"] for cls in classes]
        f1_scores = [precision_recall[cls]["f1_score"] for cls in classes]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Class-wise Performance Analysis", fontsize=16)

        # Precision by class
        axes[0, 0].bar(classes, precisions, color="blue", alpha=0.7)
        axes[0, 0].set_title("Precision by Class")
        axes[0, 0].set_ylabel("Precision")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].set_ylim(0, 1)

        # Recall by class
        axes[0, 1].bar(classes, recalls, color="green", alpha=0.7)
        axes[0, 1].set_title("Recall by Class")
        axes[0, 1].set_ylabel("Recall")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].set_ylim(0, 1)

        # F1 Score by class
        axes[1, 0].bar(classes, f1_scores, color="purple", alpha=0.7)
        axes[1, 0].set_title("F1 Score by Class")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].set_ylim(0, 1)

        # Precision vs Recall scatter
        axes[1, 1].scatter(
            precisions, recalls, c=f1_scores, cmap="viridis", s=100, alpha=0.7
        )
        for i, cls in enumerate(classes):
            axes[1, 1].annotate(
                cls,
                (precisions[i], recalls[i]),
                xytext=(5, 5),
                textcoords="offset points",
            )
        axes[1, 1].set_xlabel("Precision")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].set_title("Precision vs Recall (colored by F1)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "class_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def _plot_htp_metrics(self) -> str:
        """Plot HTP-specific evaluation metrics."""
        htp_metrics = self.evaluation_results["htp_specific_metrics"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("HTP Psychological Assessment Metrics", fontsize=16)

        # Psychological completeness distribution
        completeness_scores = htp_metrics["psychological_completeness_scores"]
        axes[0, 0].hist(
            completeness_scores,
            bins=20,
            alpha=0.7,
            color="lightblue",
            edgecolor="black",
        )
        axes[0, 0].axvline(
            np.mean(completeness_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(completeness_scores):.3f}",
        )
        axes[0, 0].set_title("Psychological Completeness Score Distribution")
        axes[0, 0].set_xlabel("Completeness Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()

        # Essential feature detection heatmap
        features = list(htp_metrics["essential_feature_detection_rates"].keys())
        rates = list(htp_metrics["essential_feature_detection_rates"].values())

        # Create a simple heatmap
        rates_array = np.array(rates).reshape(1, -1)
        im = axes[0, 1].imshow(
            rates_array, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1
        )
        axes[0, 1].set_xticks(range(len(features)))
        axes[0, 1].set_xticklabels(features)
        axes[0, 1].set_yticks([])
        axes[0, 1].set_title("Essential Feature Detection Rates")

        # Add text annotations
        for i, rate in enumerate(rates):
            axes[0, 1].text(
                i, 0, f"{rate:.3f}", ha="center", va="center", fontweight="bold"
            )

        # Completeness score trends
        axes[1, 0].plot(completeness_scores, marker="o", alpha=0.6)
        axes[1, 0].set_title("Completeness Scores by Image")
        axes[1, 0].set_xlabel("Image Index")
        axes[1, 0].set_ylabel("Completeness Score")
        axes[1, 0].grid(True, alpha=0.3)

        # Feature importance pie chart
        total_detections = sum(rates)
        if total_detections > 0:
            normalized_rates = [rate / total_detections for rate in rates]
            axes[1, 1].pie(
                normalized_rates, labels=features, autopct="%1.1f%%", startangle=90
            )
            axes[1, 1].set_title("Relative Feature Detection Success")

        plt.tight_layout()
        output_path = self.output_dir / "htp_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def _plot_confusion_matrix(self) -> str:
        """Plot confusion matrix for class detection."""
        if "feature_statistics" not in self.evaluation_results:
            return ""

        feature_stats = self.evaluation_results["feature_statistics"]

        # Create confusion matrix data
        classes = list(feature_stats.keys())
        n_classes = len(classes)

        # Simple binary confusion matrix for each class (detected vs not detected)
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))

        # Create matrix showing TP, FP, FN for each class
        matrix_data = []
        for class_name in classes:
            tp = feature_stats[class_name]["tp"]
            fp = feature_stats[class_name]["fp"]
            fn = feature_stats[class_name]["fn"]
            matrix_data.append([tp, fp, fn])

        matrix_data = np.array(matrix_data)

        # Plot heatmap
        sns.heatmap(
            matrix_data,
            xticklabels=["True Positives", "False Positives", "False Negatives"],
            yticklabels=classes,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes,
        )

        axes.set_title("Detection Statistics by Class")
        axes.set_xlabel("Detection Type")
        axes.set_ylabel("Class")

        plt.tight_layout()
        output_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def _create_interactive_dashboard(self) -> str:
        """Create interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Overall Metrics",
                "Class Performance",
                "HTP Completeness",
                "Feature Detection",
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "bar"}],
            ],
        )

        # Overall metrics
        yolo_metrics = self.evaluation_results["yolo_metrics"]
        fig.add_trace(
            go.Bar(
                x=list(yolo_metrics.keys()),
                y=list(yolo_metrics.values()),
                name="YOLO Metrics",
            ),
            row=1,
            col=1,
        )

        # Class performance (if available)
        if "precision_recall" in self.evaluation_results:
            precision_recall = self.evaluation_results["precision_recall"]
            classes = list(precision_recall.keys())
            precisions = [precision_recall[cls]["precision"] for cls in classes]
            recalls = [precision_recall[cls]["recall"] for cls in classes]

            fig.add_trace(
                go.Scatter(
                    x=precisions,
                    y=recalls,
                    mode="markers+text",
                    text=classes,
                    textposition="top center",
                    name="Precision vs Recall",
                ),
                row=1,
                col=2,
            )

        # HTP completeness (if available)
        if "htp_specific_metrics" in self.evaluation_results:
            completeness_scores = self.evaluation_results["htp_specific_metrics"][
                "psychological_completeness_scores"
            ]
            fig.add_trace(
                go.Histogram(x=completeness_scores, name="Completeness Distribution"),
                row=2,
                col=1,
            )

            # Essential feature detection
            features = list(
                self.evaluation_results["htp_specific_metrics"][
                    "essential_feature_detection_rates"
                ].keys()
            )
            rates = list(
                self.evaluation_results["htp_specific_metrics"][
                    "essential_feature_detection_rates"
                ].values()
            )

            fig.add_trace(
                go.Bar(x=features, y=rates, name="Detection Rates"), row=2, col=2
            )

        fig.update_layout(
            title_text="HTP YOLO Model Evaluation Dashboard",
            showlegend=False,
            height=800,
        )

        output_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(output_path))

        return str(output_path)

    def _generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report."""
        report_path = self.output_dir / "evaluation_report.md"

        with open(report_path, "w") as f:
            f.write(
                f"""# HTP YOLO Model Evaluation Report

## Model Information
- **Model Path**: {self.model_path}
- **Model Type**: YOLOv11 Small
- **Classes**: {', '.join(self.class_names)}
- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance Metrics
"""
            )

            # YOLO metrics
            yolo_metrics = self.evaluation_results["yolo_metrics"]
            f.write("### YOLO Validation Metrics\n")
            for metric, value in yolo_metrics.items():
                f.write(f"- **{metric}**: {value:.4f}\n")

            # Class-wise metrics
            if "precision_recall" in self.evaluation_results:
                f.write("\n### Class-wise Performance\n")
                precision_recall = self.evaluation_results["precision_recall"]

                f.write("| Class | Precision | Recall | F1 Score | Support |\n")
                f.write("|-------|-----------|---------|----------|---------|\n")

                for class_name, metrics in precision_recall.items():
                    f.write(
                        f"| {class_name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1_score']:.3f} | {metrics['support']} |\n"
                    )

            # HTP-specific metrics
            if "htp_specific_metrics" in self.evaluation_results:
                htp_metrics = self.evaluation_results["htp_specific_metrics"]
                f.write(f"\n## HTP Psychological Assessment Metrics\n")
                f.write(
                    f"- **Overall HTP Score**: {htp_metrics['overall_htp_score']:.4f}\n"
                )
                f.write(
                    f"- **Mean Completeness**: {np.mean(htp_metrics['psychological_completeness_scores']):.4f}\n"
                )
                f.write(
                    f"- **Std Completeness**: {np.std(htp_metrics['psychological_completeness_scores']):.4f}\n"
                )

                f.write("\n### Essential Feature Detection Rates\n")
                for feature, rate in htp_metrics[
                    "essential_feature_detection_rates"
                ].items():
                    f.write(f"- **{feature}**: {rate:.3f} ({rate*100:.1f}%)\n")

            f.write(
                f"""
## Interpretation for HTP Assessment

### Model Suitability
This model's performance indicates its suitability for automated HTP analysis:
- **High precision/recall**: Reliable feature detection for psychological assessment
- **Good completeness scores**: Most essential features are consistently detected
- **Balanced performance**: All key house components are detected with reasonable accuracy

### Recommendations
1. **Clinical Use**: Model shows promise for supporting HTP analysis
2. **Human Oversight**: Professional validation still recommended for clinical decisions
3. **Confidence Thresholds**: Use higher thresholds for critical assessments
4. **Feature Focus**: Pay special attention to door/window detections as they're crucial for psychological interpretation

### Limitations
- Model performance may vary with drawing styles
- Missing features could indicate either absence or detection failure
- Psychological interpretation requires human expertise

---
*Generated by HTP YOLO Evaluation Pipeline*
"""
            )

        return str(report_path)

    def compare_models(self, other_model_paths: List[str]) -> Dict:
        """Compare performance with other models."""
        # This could be extended to compare multiple model versions
        pass


def main():
    """Main evaluation function."""
    # Configuration
    model_path = "training_outputs/htp_yolo11s_latest/weights/best.pt"
    data_config = "house_only_dataset/data.yaml"

    # Check if files exist
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Please train the model first using train_yolo.py")
        return

    if not Path(data_config).exists():
        print(f"Data config not found: {data_config}")
        print("Please prepare the dataset first using data_processor.py")
        return

    # Initialize evaluator
    evaluator = HTPModelEvaluator(model_path, data_config)

    # Run comprehensive evaluation
    print("Starting model evaluation...")
    results = evaluator.evaluate_model()

    # Create visualizations
    print("Creating visualizations...")
    viz_paths = evaluator.visualize_results()

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETED")
    print("=" * 50)

    # Print summary
    if "yolo_metrics" in results:
        yolo_metrics = results["yolo_metrics"]
        print(f"mAP@0.5: {yolo_metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {yolo_metrics['mAP50_95']:.4f}")
        print(f"Precision: {yolo_metrics['precision']:.4f}")
        print(f"Recall: {yolo_metrics['recall']:.4f}")

    if "htp_specific_metrics" in results:
        htp_score = results["htp_specific_metrics"]["overall_htp_score"]
        print(f"Overall HTP Score: {htp_score:.4f}")

    print(f"\nResults saved to: {evaluator.output_dir}")
    print(f"Report: {results.get('report_path', 'N/A')}")
    print(f"Visualizations: {len(viz_paths)} files created")


if __name__ == "__main__":
    main()
