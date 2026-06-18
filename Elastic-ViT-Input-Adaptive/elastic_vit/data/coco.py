from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection


COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class CocoMultilabelDataset(Dataset):
    def __init__(self, root: str | Path, split: str, transform: Callable | None = None) -> None:
        super().__init__()
        root_path = Path(root)
        image_dir = root_path / ("train2017" if split == "train" else "val2017")
        annotation_file = root_path / "annotations" / f"instances_{'train2017' if split == 'train' else 'val2017'}.json"
        self.dataset = CocoDetection(root=str(image_dir), annFile=str(annotation_file), transform=None)
        self.transform = transform
        categories = self.dataset.coco.loadCats(self.dataset.coco.getCatIds())
        self.category_to_index: Dict[int, int] = {
            category["id"]: index for index, category in enumerate(sorted(categories, key=lambda item: item["id"]))
        }
        if len(self.category_to_index) != len(COCO_CLASS_NAMES):
            raise ValueError("Expected 80 COCO categories.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, annotations = self.dataset[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.zeros(len(self.category_to_index), dtype=torch.float32)
        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id in self.category_to_index:
                target[self.category_to_index[category_id]] = 1.0
        return image, target
