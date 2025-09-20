"""
Data preprocessing script for HTP (House-Tree-Person) dataset.
Filters and prepares data specifically for house feature detection and analysis.
"""

import yaml
import cv2
from pathlib import Path
from typing import Dict, List, Optional
import shutil
from tqdm import tqdm


class HTPDataProcessor:
    """Process HTP dataset for house feature detection."""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.class_names = ["chimney", "door", "house", "roof", "sun", "wall", "window"]
        self.house_classes = [
            "chimney",
            "door",
            "house",
            "roof",
            "wall",
            "window",
        ]  # Exclude 'sun'
        self.house_class_ids = [0, 1, 2, 3, 5, 6]  # Corresponding IDs

        # Load original data config
        with open(self.data_root / "data.yaml", "r") as f:
            self.original_config = yaml.safe_load(f)

    def filter_house_only_dataset(self, output_dir: str = None):
        """Create a filtered dataset containing only house-related classes."""
        if output_dir is None:
            # Default to data/processed/house_only_dataset relative to project root
            project_root = Path(__file__).parent.parent
            output_path = project_root / "data" / "processed" / "house_only_dataset"
        else:
            output_path = Path(output_dir)

        # Ensure parent directories exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(exist_ok=True)

        # Create directory structure
        for split in ["train", "valid", "test"]:
            (output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

        stats = {"train": 0, "valid": 0, "test": 0}

        for split in ["train", "valid", "test"]:
            print(f"Processing {split} split...")

            images_dir = self.data_root / split / "images"
            labels_dir = self.data_root / split / "labels"

            output_images_dir = output_path / split / "images"
            output_labels_dir = output_path / split / "labels"

            if not images_dir.exists():
                print(f"Warning: {images_dir} not found, skipping {split}")
                continue

            image_files = list(images_dir.glob("*"))

            for img_file in tqdm(image_files, desc=f"Processing {split}"):
                # Find corresponding label file
                label_file = labels_dir / (img_file.stem + ".txt")

                if not label_file.exists():
                    continue

                # Read and filter labels
                filtered_labels = self._filter_house_labels(label_file)

                # Only keep images that have house-related objects
                if filtered_labels:
                    # Copy image
                    shutil.copy2(img_file, output_images_dir / img_file.name)

                    # Write filtered labels
                    with open(output_labels_dir / label_file.name, "w") as f:
                        f.write("\n".join(filtered_labels))

                    stats[split] += 1

        # Create new data.yaml
        new_config = {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(self.house_classes),
            "names": self.house_classes,
        }

        with open(output_path / "data.yaml", "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        print(f"Dataset filtering complete!")
        print(f"Statistics: {stats}")
        return output_path, stats

    def _filter_house_labels(self, label_file: Path) -> List[str]:
        """Filter labels to keep only house-related classes."""
        filtered_lines = []

        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                class_id = int(parts[0])

                # Check if this is a house-related class
                if class_id in self.house_class_ids:
                    # Remap class ID to new index
                    new_class_id = self.house_class_ids.index(class_id)
                    new_line = f"{new_class_id} " + " ".join(parts[1:])
                    filtered_lines.append(new_line)

        return filtered_lines

    def analyze_dataset_statistics(self, dataset_path: Optional[str] = None) -> Dict:
        """Analyze dataset statistics for house features."""
        if dataset_path is None:
            dataset_path = self.data_root
        else:
            dataset_path = Path(dataset_path)

        stats = {
            "splits": {},
            "class_distribution": {name: 0 for name in self.house_classes},
            "image_sizes": [],
            "annotations_per_image": [],
        }

        for split in ["train", "valid", "test"]:
            split_stats = {
                "images": 0,
                "annotations": 0,
                "classes": {name: 0 for name in self.house_classes},
            }

            images_dir = dataset_path / split / "images"
            labels_dir = dataset_path / split / "labels"

            if not images_dir.exists():
                continue

            image_files = list(images_dir.glob("*"))
            split_stats["images"] = len(image_files)

            for img_file in image_files:
                # Get image dimensions
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    stats["image_sizes"].append((w, h))

                # Count annotations
                label_file = labels_dir / (img_file.stem + ".txt")
                if label_file.exists():
                    with open(label_file, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        split_stats["annotations"] += len(lines)
                        stats["annotations_per_image"].append(len(lines))

                        for line in lines:
                            class_id = int(line.split()[0])
                            if class_id < len(self.house_classes):
                                class_name = self.house_classes[class_id]
                                split_stats["classes"][class_name] += 1
                                stats["class_distribution"][class_name] += 1

            stats["splits"][split] = split_stats

        return stats


def main():
    """Main function to demonstrate data processing."""
    # Initialize processor
    processor = HTPDataProcessor(data_root=".")

    # Filter dataset for house-only classes
    print("Creating house-only dataset...")
    house_dataset_path, stats = processor.filter_house_only_dataset()
    print(f"House-only dataset created at: {house_dataset_path}")

    # Analyze dataset statistics
    print("\nAnalyzing dataset statistics...")
    dataset_stats = processor.analyze_dataset_statistics(house_dataset_path)

    print("\nDataset Statistics:")
    for split, split_stats in dataset_stats["splits"].items():
        print(
            f"{split}: {split_stats['images']} images, {split_stats['annotations']} annotations"
        )

    print("\nClass Distribution:")
    for class_name, count in dataset_stats["class_distribution"].items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
