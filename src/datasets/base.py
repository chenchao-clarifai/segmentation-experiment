import os

import torch
from PIL import Image
from torchvision.datasets import VisionDataset

EXT = ["jpg", "jpeg", "png", "tiff", "tif"]

__all__ = ["SemanticSegmentationBase"]


def is_image(filename):
    ext = filename.split(".")[-1].lower()
    return ext in EXT


class SemanticSegmentationBase(VisionDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self, image_root: str, mask_root: str, transforms=None):
        super().__init__(image_root, transforms=transforms)

        if isinstance(mask_root, torch._six.string_classes):
            mask_root = os.path.expanduser(mask_root)
        self.image_root = image_root
        self.mask_root = mask_root

        list_of_images = sorted([p for p in os.listdir(image_root) if is_image(p)])
        list_of_masks = sorted([p for p in os.listdir(mask_root) if is_image(p)])

        for img_fn, msk_fn in zip(list_of_images, list_of_masks):
            assert (
                img_fn.split(".")[0] == msk_fn.split(".")[0]
            ), f"Image filename: {img_fn} and mask filename: {msk_fn} mismatched."

        self.list_of_images = list_of_images
        self.list_of_masks = list_of_masks

        self.num_images = len(list_of_images)

        self.transforms = transforms

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.image_root, self.list_of_images[idx]))
        msk = Image.open(os.path.join(self.mask_root, self.list_of_masks[idx]))

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
