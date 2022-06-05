import os
from typing import Optional

import torch
from PIL import Image
from torchvision.datasets import VisionDataset

EXT = ["jpg", "jpeg", "png", "tiff", "tif"]

__all__ = ["SemanticSegmentationDataset"]


def is_image(filename: str, extension: Optional[str] = None) -> bool:
    ext = filename.split(".")[-1].lower()
    if not extension:
        return ext in EXT
    return ext == extension.lower()


class SemanticSegmentationDataset(VisionDataset):
    """Base Class for Semantic Segmentation Dataset."""

    def __init__(
        self,
        image_root: str,
        mask_root: Optional[str] = None,
        transforms=None,
        image_ext="jpg",
        mask_ext="png",
    ):
        super().__init__(image_root, transforms=transforms)

        if mask_root is None:
            mask_root = image_root
        if isinstance(mask_root, torch._six.string_classes):
            mask_root = os.path.expanduser(mask_root)

        self.image_ext = image_ext.strip().lower()
        self.mask_ext = mask_ext.strip().lower()
        self.image_root = image_root
        self.mask_root = mask_root

        list_of_images = sorted(
            [p for p in os.listdir(image_root) if is_image(p, image_ext)]
        )
        list_of_masks = sorted(
            [p for p in os.listdir(mask_root) if is_image(p, mask_ext)]
        )

        for img_fn, msk_fn in zip(list_of_images, list_of_masks):
            assert (
                img_fn.split(".")[0] == msk_fn.split(".")[0]
            ), f"Image filename: {img_fn} and mask filename: {msk_fn} mismatched."

        self.list_of_images = list_of_images
        self.list_of_masks = list_of_masks

        self.num_images = len(list_of_images)

        self.transforms = transforms

    def __getitem__(self, idx):

        img = Image.open(
            os.path.join(self.image_root, self.list_of_images[idx])
        ).convert("RGB")
        msk = Image.open(os.path.join(self.mask_root, self.list_of_masks[idx])).convert(
            "L"
        )

        if self.transforms:
            if isinstance(self.transforms, (list, tuple)):
                out = {"image": img, "mask": msk}
                for T in self.transforms:
                    out = T(**out)
            else:
                out = self.transforms(image=img, mask=msk)
            img, msk = out["image"], out["mask"]

        return {"image": img, "mask": msk}

    def __len__(self):
        return self.num_images
