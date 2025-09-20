"""
HTP Feature Extraction and Analysis Pipeline
Analyzes detected house features for psychological assessment according to HTP guidelines.
"""

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm


@dataclass
class HTPFeatureAnalysis:
    """Data class for HTP feature analysis results."""

    image_path: str
    image_size: Tuple[int, int]
    detected_features: List[str]
    missing_features: List[str]

    # Size analysis
    house_size_category: str  # 'small', 'normal', 'large'
    house_area_ratio: float

    # Position analysis
    house_placement: List[str]  # ['left/center/right', 'high/middle/low']

    # Feature characteristics
    door_present: bool
    door_characteristics: Dict[str, Any]
    window_count: int
    window_characteristics: Dict[str, Any]
    chimney_present: bool
    chimney_characteristics: Dict[str, Any]
    wall_characteristics: Dict[str, Any]
    roof_characteristics: Dict[str, Any]

    # Psychological interpretations
    psychological_indicators: Dict[str, List[str]]
    risk_factors: List[str]
    positive_indicators: List[str]

    # Confidence scores
    detection_confidence: Dict[str, float]
    analysis_timestamp: str


class HTPFeatureExtractor:
    """Extract and analyze HTP features from house drawings using trained YOLO model."""

    def __init__(self, model_path: str):
        """
        Initialize feature extractor.

        Args:
            model_path: Path to trained YOLO model
        """
        self.model = YOLO(model_path)
        self.class_names = ["chimney", "door", "house", "roof", "wall", "window"]

        # HTP interpretation guidelines
        self.htp_guidelines = {
            "house_size": {
                "small": {
                    "threshold": 0.1,
                    "interpretation": [
                        "withdrawal",
                        "feelings of inadequacy",
                        "rejection of home life",
                    ],
                },
                "large": {
                    "threshold": 0.6,
                    "interpretation": [
                        "frustration",
                        "hostility",
                        "aggressive tendencies",
                    ],
                },
            },
            "placement": {
                "left": [
                    "impulsivity",
                    "emotional satisfaction orientation",
                    "focus on past",
                ],
                "right": ["intellectualization", "focus on present reality"],
                "center": ["reasonably secure individual"],
                "high": ["high ambition", "active fantasy life", "optimism"],
                "low": ["insecurity", "low self-esteem", "depressive tendencies"],
            },
            "door": {
                "missing": ["insecurity", "difficulty connecting with others"],
                "tiny": ["fearfulness", "timidity", "reluctance to engage"],
                "large": ["dependency", "emotional vulnerability"],
                "closed": ["guarded personality", "lack of social interaction"],
            },
            "window": {
                "missing": ["guarded", "suspicious", "hostile"],
                "excessive": ["exhibitionism", "openness to environment"],
                "barred": ["withdrawal", "defensiveness"],
            },
            "chimney": {
                "missing": ["lack of warmth in home environment"],
                "excessive_smoke": ["inner tension", "emotional distress"],
            },
        }

    def analyze_image(
        self, image_path: str, confidence_threshold: float = 0.25
    ) -> HTPFeatureAnalysis:
        """
        Analyze a single house drawing for HTP features.

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for detections

        Returns:
            HTPFeatureAnalysis object with complete analysis
        """
        # Load and analyze image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        h, w = image.shape[:2]

        # Run YOLO inference
        results = self.model(image_path, conf=confidence_threshold)

        # Extract detections
        detections = self._extract_detections(results[0], w, h)

        # Perform HTP analysis
        analysis = self._perform_htp_analysis(detections, w, h, image_path)

        return analysis

    def _extract_detections(self, result, img_width: int, img_height: int) -> Dict:
        """Extract detection information from YOLO results."""
        detections = {
            "boxes": {},
            "masks": {},
            "confidences": {},
            "detected_classes": set(),
        }

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls)
                confidence = float(box.conf)

                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    detections["detected_classes"].add(class_name)

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections["boxes"][class_name] = {
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "area": (x2 - x1) * (y2 - y1),
                    }
                    detections["confidences"][class_name] = confidence

        # Handle segmentation masks if available
        if result.masks is not None:
            for i, mask in enumerate(result.masks):
                if i < len(result.boxes):
                    class_id = int(result.boxes[i].cls)
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                        detections["masks"][class_name] = mask.data.cpu().numpy()

        return detections

    def _perform_htp_analysis(
        self, detections: Dict, img_width: int, img_height: int, image_path: str
    ) -> HTPFeatureAnalysis:
        """Perform comprehensive HTP psychological analysis."""

        detected_features = list(detections["detected_classes"])
        missing_features = [
            cls for cls in self.class_names if cls not in detected_features
        ]

        # Initialize analysis components
        house_size_category = "unknown"
        house_area_ratio = 0.0
        house_placement = []
        psychological_indicators = {}
        risk_factors = []
        positive_indicators = []

        # Analyze house size and placement
        if "house" in detections["boxes"]:
            house_info = detections["boxes"]["house"]
            house_area = house_info["area"]
            image_area = img_width * img_height
            house_area_ratio = house_area / image_area

            # Categorize house size
            if (
                house_area_ratio
                < self.htp_guidelines["house_size"]["small"]["threshold"]
            ):
                house_size_category = "small"
                psychological_indicators["house_size"] = self.htp_guidelines[
                    "house_size"
                ]["small"]["interpretation"]
                risk_factors.extend(["withdrawal", "inadequacy feelings"])
            elif (
                house_area_ratio
                > self.htp_guidelines["house_size"]["large"]["threshold"]
            ):
                house_size_category = "large"
                psychological_indicators["house_size"] = self.htp_guidelines[
                    "house_size"
                ]["large"]["interpretation"]
                risk_factors.extend(["hostility", "aggression"])
            else:
                house_size_category = "normal"
                positive_indicators.append("appropriate size perception")

            # Analyze placement
            center_x, center_y = house_info["center"]

            # Horizontal placement
            if center_x < img_width * 0.3:
                house_placement.append("left")
                psychological_indicators["placement_horizontal"] = self.htp_guidelines[
                    "placement"
                ]["left"]
            elif center_x > img_width * 0.7:
                house_placement.append("right")
                psychological_indicators["placement_horizontal"] = self.htp_guidelines[
                    "placement"
                ]["right"]
            else:
                house_placement.append("center")
                psychological_indicators["placement_horizontal"] = self.htp_guidelines[
                    "placement"
                ]["center"]
                positive_indicators.append("balanced horizontal placement")

            # Vertical placement
            if center_y < img_height * 0.3:
                house_placement.append("high")
                psychological_indicators["placement_vertical"] = self.htp_guidelines[
                    "placement"
                ]["high"]
                positive_indicators.append("optimistic tendencies")
            elif center_y > img_height * 0.7:
                house_placement.append("low")
                psychological_indicators["placement_vertical"] = self.htp_guidelines[
                    "placement"
                ]["low"]
                risk_factors.extend(["insecurity", "low self-esteem"])
            else:
                house_placement.append("middle")
                positive_indicators.append("balanced vertical placement")

        # Analyze door characteristics
        door_present = "door" in detected_features
        door_characteristics = self._analyze_door(detections, img_width, img_height)
        if not door_present:
            psychological_indicators["door_missing"] = self.htp_guidelines["door"][
                "missing"
            ]
            risk_factors.extend(["social difficulties", "isolation"])

        # Analyze window characteristics
        window_count = len([k for k in detections["boxes"].keys() if k == "window"])
        window_characteristics = self._analyze_windows(detections, window_count)

        # Analyze chimney
        chimney_present = "chimney" in detected_features
        chimney_characteristics = self._analyze_chimney(detections)
        if not chimney_present:
            psychological_indicators["chimney_missing"] = self.htp_guidelines[
                "chimney"
            ]["missing"]
            risk_factors.append("lack of emotional warmth")

        # Analyze walls and roof
        wall_characteristics = self._analyze_walls(detections)
        roof_characteristics = self._analyze_roof(detections)

        # Create final analysis
        analysis = HTPFeatureAnalysis(
            image_path=image_path,
            image_size=(img_width, img_height),
            detected_features=detected_features,
            missing_features=missing_features,
            house_size_category=house_size_category,
            house_area_ratio=house_area_ratio,
            house_placement=house_placement,
            door_present=door_present,
            door_characteristics=door_characteristics,
            window_count=window_count,
            window_characteristics=window_characteristics,
            chimney_present=chimney_present,
            chimney_characteristics=chimney_characteristics,
            wall_characteristics=wall_characteristics,
            roof_characteristics=roof_characteristics,
            psychological_indicators=psychological_indicators,
            risk_factors=list(set(risk_factors)),
            positive_indicators=list(set(positive_indicators)),
            detection_confidence=detections["confidences"],
            analysis_timestamp=datetime.now().isoformat(),
        )

        return analysis

    def _analyze_door(self, detections: Dict, img_width: int, img_height: int) -> Dict:
        """Analyze door characteristics for HTP assessment."""
        characteristics = {
            "present": "door" in detections["boxes"],
            "size_category": "unknown",
            "position": "unknown",
            "accessibility": "unknown",
        }

        if "door" in detections["boxes"]:
            door_info = detections["boxes"]["door"]
            door_area = door_info["area"]

            # Analyze relative size
            if "house" in detections["boxes"]:
                house_area = detections["boxes"]["house"]["area"]
                door_ratio = door_area / house_area if house_area > 0 else 0

                if door_ratio < 0.02:
                    characteristics["size_category"] = "tiny"
                elif door_ratio > 0.15:
                    characteristics["size_category"] = "large"
                else:
                    characteristics["size_category"] = "normal"

            # Analyze position relative to house
            door_center = door_info["center"]
            if "house" in detections["boxes"]:
                house_center = detections["boxes"]["house"]["center"]

                # Check if door is centered on house
                horizontal_offset = abs(door_center[0] - house_center[0])
                if horizontal_offset < detections["boxes"]["house"]["width"] * 0.1:
                    characteristics["position"] = "centered"
                else:
                    characteristics["position"] = "off-center"

        return characteristics

    def _analyze_windows(self, detections: Dict, window_count: int) -> Dict:
        """Analyze window characteristics."""
        characteristics = {
            "count": window_count,
            "size_variation": "unknown",
            "placement": "unknown",
            "interpretation": [],
        }

        if window_count == 0:
            characteristics["interpretation"] = self.htp_guidelines["window"]["missing"]
        elif window_count > 4:
            characteristics["interpretation"] = self.htp_guidelines["window"][
                "excessive"
            ]
        else:
            characteristics["interpretation"] = ["normal window count"]

        return characteristics

    def _analyze_chimney(self, detections: Dict) -> Dict:
        """Analyze chimney characteristics."""
        characteristics = {
            "present": "chimney" in detections["boxes"],
            "size": "unknown",
            "smoke_present": False,  # Would need additional detection
            "position": "unknown",
        }

        if "chimney" in detections["boxes"]:
            chimney_info = detections["boxes"]["chimney"]

            # Analyze size relative to house
            if "house" in detections["boxes"]:
                house_area = detections["boxes"]["house"]["area"]
                chimney_area = chimney_info["area"]
                chimney_ratio = chimney_area / house_area if house_area > 0 else 0

                if chimney_ratio < 0.01:
                    characteristics["size"] = "small"
                elif chimney_ratio > 0.05:
                    characteristics["size"] = "large"
                else:
                    characteristics["size"] = "normal"

        return characteristics

    def _analyze_walls(self, detections: Dict) -> Dict:
        """Analyze wall characteristics."""
        characteristics = {
            "present": "wall" in detections["boxes"],
            "thickness": "unknown",
            "completeness": "unknown",
        }

        # Additional analysis would require more sophisticated detection
        return characteristics

    def _analyze_roof(self, detections: Dict) -> Dict:
        """Analyze roof characteristics."""
        characteristics = {
            "present": "roof" in detections["boxes"],
            "shape": "unknown",
            "size": "unknown",
        }

        if "roof" in detections["boxes"]:
            roof_info = detections["boxes"]["roof"]

            # Analyze relative size
            if "house" in detections["boxes"]:
                house_area = detections["boxes"]["house"]["area"]
                roof_area = roof_info["area"]
                roof_ratio = roof_area / house_area if house_area > 0 else 0

                if roof_ratio < 0.1:
                    characteristics["size"] = "small"
                elif roof_ratio > 0.4:
                    characteristics["size"] = "large"
                else:
                    characteristics["size"] = "normal"

        return characteristics

    def generate_report(
        self, analysis: HTPFeatureAnalysis, output_path: str = None
    ) -> str:
        """Generate a comprehensive HTP analysis report."""
        if output_path is None:
            output_path = f"htp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        report = f"""# HTP Analysis Report

## Image Information
- **File**: {Path(analysis.image_path).name}
- **Size**: {analysis.image_size[0]} x {analysis.image_size[1]} pixels
- **Analysis Date**: {analysis.analysis_timestamp}

## Detection Summary
### Detected Features
{', '.join(analysis.detected_features) if analysis.detected_features else 'None'}

### Missing Features
{', '.join(analysis.missing_features) if analysis.missing_features else 'None'}

## Psychological Analysis

### House Characteristics
- **Size Category**: {analysis.house_size_category}
- **Area Ratio**: {analysis.house_area_ratio:.3f}
- **Placement**: {' and '.join(analysis.house_placement)}

### Feature Analysis
- **Door Present**: {analysis.door_present}
- **Window Count**: {analysis.window_count}
- **Chimney Present**: {analysis.chimney_present}

## Psychological Indicators

### Risk Factors
"""

        for factor in analysis.risk_factors:
            report += f"- {factor}\n"

        report += "\n### Positive Indicators\n"
        for indicator in analysis.positive_indicators:
            report += f"- {indicator}\n"

        report += f"""
### Detailed Interpretations
"""

        for category, indicators in analysis.psychological_indicators.items():
            report += f"\n**{category.replace('_', ' ').title()}**:\n"
            for indicator in indicators:
                report += f"- {indicator}\n"

        report += f"""
## Detection Confidence Scores
"""

        for feature, confidence in analysis.detection_confidence.items():
            report += f"- **{feature}**: {confidence:.3f}\n"

        report += f"""
---
*This analysis is based on automated detection and should be interpreted by qualified professionals.*
"""

        with open(output_path, "w") as f:
            f.write(report)

        return output_path

    def visualize_analysis(
        self, analysis: HTPFeatureAnalysis, output_path: str = None
    ) -> str:
        """Create visualization of the analysis results."""
        if output_path is None:
            output_path = (
                f"htp_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

        # Load original image
        image = cv2.imread(analysis.image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"HTP Analysis: {Path(analysis.image_path).name}", fontsize=16)

        # Original image with detections
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Feature presence chart
        all_features = self.class_names
        feature_status = [
            "Present" if f in analysis.detected_features else "Missing"
            for f in all_features
        ]
        colors = [
            "green" if status == "Present" else "red" for status in feature_status
        ]

        axes[0, 1].barh(all_features, [1] * len(all_features), color=colors, alpha=0.7)
        axes[0, 1].set_title("Feature Detection Status")
        axes[0, 1].set_xlabel("Detection Status")

        # Risk factors vs positive indicators
        risk_count = len(analysis.risk_factors)
        positive_count = len(analysis.positive_indicators)

        axes[1, 0].pie(
            [risk_count, positive_count],
            labels=["Risk Factors", "Positive Indicators"],
            colors=["red", "green"],
            autopct="%1.1f%%",
        )
        axes[1, 0].set_title("Psychological Indicators Balance")

        # Confidence scores
        if analysis.detection_confidence:
            features = list(analysis.detection_confidence.keys())
            confidences = list(analysis.detection_confidence.values())

            axes[1, 1].bar(features, confidences, color="blue", alpha=0.7)
            axes[1, 1].set_title("Detection Confidence Scores")
            axes[1, 1].set_ylabel("Confidence")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def batch_analysis(
        self, image_directory: str, output_directory: str = "htp_batch_analysis"
    ) -> pd.DataFrame:
        """Perform batch analysis on multiple images."""
        image_dir = Path(image_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True)

        # Find all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))

        results = []

        for image_file in tqdm(image_files, desc="Analyzing images"):
            try:
                # Analyze image
                analysis = self.analyze_image(str(image_file))

                # Generate individual report
                report_path = output_dir / f"{image_file.stem}_report.md"
                self.generate_report(analysis, str(report_path))

                # Generate visualization
                viz_path = output_dir / f"{image_file.stem}_visualization.png"
                self.visualize_analysis(analysis, str(viz_path))

                # Collect data for summary
                result_data = {
                    "image_name": image_file.name,
                    "house_size_category": analysis.house_size_category,
                    "house_area_ratio": analysis.house_area_ratio,
                    "door_present": analysis.door_present,
                    "window_count": analysis.window_count,
                    "chimney_present": analysis.chimney_present,
                    "risk_factor_count": len(analysis.risk_factors),
                    "positive_indicator_count": len(analysis.positive_indicators),
                    "detected_feature_count": len(analysis.detected_features),
                }

                results.append(result_data)

            except Exception as e:
                print(f"Error analyzing {image_file}: {e}")
                continue

        # Create summary DataFrame
        df = pd.DataFrame(results)

        # Save summary
        summary_path = output_dir / "batch_analysis_summary.csv"
        df.to_csv(summary_path, index=False)

        # Create summary visualization
        self._create_batch_summary_plots(df, output_dir)

        print(f"Batch analysis complete. Results saved to: {output_dir}")
        return df

    def _create_batch_summary_plots(self, df: pd.DataFrame, output_dir: Path):
        """Create summary plots for batch analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("HTP Batch Analysis Summary", fontsize=16)

        # House size distribution
        size_counts = df["house_size_category"].value_counts()
        axes[0, 0].pie(size_counts.values, labels=size_counts.index, autopct="%1.1f%%")
        axes[0, 0].set_title("House Size Distribution")

        # Feature presence
        feature_presence = {
            "Door": df["door_present"].sum(),
            "Chimney": df["chimney_present"].sum(),
            "No Door": (~df["door_present"]).sum(),
            "No Chimney": (~df["chimney_present"]).sum(),
        }

        axes[0, 1].bar(
            feature_presence.keys(),
            feature_presence.values(),
            color=["green", "green", "red", "red"],
        )
        axes[0, 1].set_title("Feature Presence Summary")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Window count distribution
        axes[0, 2].hist(df["window_count"], bins=10, alpha=0.7, color="blue")
        axes[0, 2].set_title("Window Count Distribution")
        axes[0, 2].set_xlabel("Number of Windows")
        axes[0, 2].set_ylabel("Frequency")

        # Risk vs positive indicators
        axes[1, 0].scatter(
            df["risk_factor_count"], df["positive_indicator_count"], alpha=0.6
        )
        axes[1, 0].set_xlabel("Risk Factor Count")
        axes[1, 0].set_ylabel("Positive Indicator Count")
        axes[1, 0].set_title("Risk vs Positive Indicators")

        # House area ratio distribution
        axes[1, 1].hist(df["house_area_ratio"], bins=20, alpha=0.7, color="orange")
        axes[1, 1].set_title("House Size Ratio Distribution")
        axes[1, 1].set_xlabel("Area Ratio")
        axes[1, 1].set_ylabel("Frequency")

        # Feature detection success
        axes[1, 2].hist(df["detected_feature_count"], bins=7, alpha=0.7, color="purple")
        axes[1, 2].set_title("Detected Feature Count")
        axes[1, 2].set_xlabel("Number of Detected Features")
        axes[1, 2].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(
            output_dir / "batch_summary_plots.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    """Main function to demonstrate feature extraction."""
    # Example usage
    model_path = (
        "training_outputs/htp_yolo11s_latest/weights/best.pt"  # Update with actual path
    )

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Please train the model first using train_yolo.py")
        return

    # Initialize extractor
    extractor = HTPFeatureExtractor(model_path)

    # Example single image analysis
    test_image = "test/images"  # Update with actual test image
    test_images = list(Path(test_image).glob("*.jpg"))

    if test_images:
        print(f"Analyzing sample image: {test_images[0]}")

        # Analyze single image
        analysis = extractor.analyze_image(str(test_images[0]))

        # Generate report
        report_path = extractor.generate_report(analysis)
        print(f"Report generated: {report_path}")

        # Create visualization
        viz_path = extractor.visualize_analysis(analysis)
        print(f"Visualization created: {viz_path}")

        # Print summary
        print(f"\nAnalysis Summary:")
        print(f"House size: {analysis.house_size_category}")
        print(f"Detected features: {', '.join(analysis.detected_features)}")
        print(f"Risk factors: {len(analysis.risk_factors)}")
        print(f"Positive indicators: {len(analysis.positive_indicators)}")

    # Batch analysis example
    if len(test_images) > 1:
        print(f"\nPerforming batch analysis on {len(test_images)} images...")
        batch_df = extractor.batch_analysis(test_image)
        print(f"Batch analysis complete. Processed {len(batch_df)} images.")


if __name__ == "__main__":
    main()
