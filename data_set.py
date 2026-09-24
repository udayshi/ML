"""Copy a small labelled image dataset into the CNN directory structure."""

from __future__ import annotations

import shutil
from pathlib import Path

SOURCE_DIRECTORY = Path("../datasets/set")
DATASET_DIRECTORY = Path("datasets")
CLASS_NAMES = ("Cat", "Dog")
TRAINING_COUNT = 100
TEST_COUNT = 20
SINGLE_PREDICTION_COUNT = 10
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def get_image_files(source_directory: Path) -> list[Path]:
    """Return source image files in deterministic order."""
    return sorted(
        path
        for path in source_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def copy_files(files: list[Path], target_directory: Path) -> None:
    """Copy image files into a target directory with sequential names."""
    target_directory.mkdir(parents=True, exist_ok=True)
    for sequence_number, source_file in enumerate(files, start=1):
        target_file = target_directory / f"{sequence_number}{source_file.suffix.lower()}"
        shutil.copy2(source_file, target_file)


def copy_class_dataset(class_name: str) -> None:
    """Copy one class into the training, test, and prediction folders."""
    source_directory = SOURCE_DIRECTORY / class_name
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Source class directory does not exist: {source_directory}")

    image_files = get_image_files(source_directory)
    required_count = TRAINING_COUNT + TEST_COUNT + SINGLE_PREDICTION_COUNT
    if len(image_files) < required_count:
        raise ValueError(
            f"{source_directory} contains {len(image_files)} images; "
            f"{required_count} are required."
        )

    training_end = TRAINING_COUNT
    test_end = training_end + TEST_COUNT
    training_files = image_files[:training_end]
    test_files = image_files[training_end:test_end]
    single_prediction_files = image_files[test_end:required_count]

    copy_files(training_files, DATASET_DIRECTORY / "training_set" / class_name)
    copy_files(test_files, DATASET_DIRECTORY / "test_set" / class_name)
    copy_files(
        single_prediction_files,
        DATASET_DIRECTORY / "single_prediction" / class_name,
    )

    print(
        f"{class_name}: copied {len(training_files)} training, "
        f"{len(test_files)} test, and "
        f"{len(single_prediction_files)} single-prediction images"
    )


def main() -> None:
    """Copy the requested number of images for every configured class."""
    if not SOURCE_DIRECTORY.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {SOURCE_DIRECTORY}")

    for class_name in CLASS_NAMES:
        copy_class_dataset(class_name)

    print(f"Dataset copied to: {DATASET_DIRECTORY}")


if __name__ == "__main__":
    main()
