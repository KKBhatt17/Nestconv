from __future__ import annotations

from torchvision import transforms

# BLIP image normalization (matches the BLIP-initialized vision tower).
BLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_image_transform(image_size: int, train: bool):
    normalize = transforms.Normalize(mean=BLIP_MEAN, std=BLIP_STD)
    if train:
        # Light augmentation only: aggressive crops/flips can invalidate the
        # answer (e.g. left/right or position questions), so we keep it tame.
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
