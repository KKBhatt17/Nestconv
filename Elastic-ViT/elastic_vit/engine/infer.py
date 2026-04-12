from __future__ import annotations

from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from elastic_vit.data.entropy_cache import compute_mean_entropy


def build_inference_transform(image_size: int):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


@torch.no_grad()
def route_single_image(
    image: Image.Image,
    model,
    router,
    device: torch.device,
    dataset_cfg: Dict,
    export_subnetwork: bool = True,
):
    image = image.convert("RGB")
    mean_entropy = compute_mean_entropy(
        image,
        image_size=dataset_cfg["image_size"],
        patch_size=dataset_cfg["patch_size"],
        num_scales=dataset_cfg["entropy_num_scales"],
    )
    entropy_tensor = torch.tensor([mean_entropy], dtype=torch.float32, device=device)
    config = router.predict_hard_config(entropy_tensor)

    tensor = build_inference_transform(dataset_cfg["image_size"])(image).unsqueeze(0).to(device)
    runtime_model = model.export_subnetwork(config).to(device) if export_subnetwork else model
    logits = runtime_model(tensor) if export_subnetwork else model(tensor, config=config)
    return logits, config, mean_entropy
