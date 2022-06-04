from typing import Any, Dict, NamedTuple

import albumentations as A
import numpy as np
import PIL
import torch
import torchvision.transforms.functional as TF

__all__ = [
    "TrainTransformsConfigs",
    "EvalTransformsConfigs",
    "get_train_transforms",
    "get_eval_transforms",
]


class TrainTransformsConfigs(NamedTuple):
    image_size: int = 512
    brightness_contrast: Dict[str, Any] = dict(p=0.5)
    gray: Dict[str, Any] = dict(p=0.2)
    blur: Dict[str, Any] = dict(p=0.1, blur_limit=5)
    compression: Dict[str, Any] = dict(p=0.1, quality_lower=10, quality_upper=50)


class EvalTransformsConfigs(NamedTuple):
    image_size: int = 512
    use_resize_p: float = 1.0
    use_random_crop_p: float = 0.0


def get_train_transforms(configs: TrainTransformsConfigs):

    a_transforms = A.Compose(
        transforms=[
            A.SmallestMaxSize(always_apply=True, max_size=configs.image_size),
            A.RandomCrop(
                always_apply=True, height=configs.image_size, width=configs.image_size
            ),
            A.RandomBrightnessContrast(**configs.brightness_contrast),
            A.ToGray(**configs.gray),
            A.MotionBlur(**configs.blur),
            A.ImageCompression(**configs.compression),
        ]
    )

    def data_pipe(image: PIL.Image, mask: PIL.Image):
        image, mask = np.array(image), np.array(mask)
        transformed = a_transforms(image=image, mask=mask)
        image = TF.to_tensor(PIL.Image.fromarray(transformed["image"]))
        mask = torch.tensor(transformed["mask"]).long()
        return dict(image=image, mask=mask)

    return data_pipe


def get_eval_transforms(configs: EvalTransformsConfigs):

    a_transforms = A.OneOf(
        transforms=[
            A.Resize(
                p=configs.use_resize_p,
                height=configs.image_size,
                width=configs.image_size,
            ),
            A.Compose(
                transforms=[
                    A.SmallestMaxSize(always_apply=True, max_size=configs.image_size),
                    A.RandomCrop(
                        always_apply=True,
                        height=configs.image_size,
                        width=configs.image_size,
                    ),
                ],
                p=configs.use_random_crop_p,
            ),
        ],
        p=1.0,
    )

    def data_pipe(image: PIL.Image, mask: PIL.Image):
        image, mask = np.array(image), np.array(mask)
        transformed = a_transforms(image=image, mask=mask)
        image = TF.to_tensor(PIL.Image.fromarray(transformed["image"]))
        mask = torch.tensor(transformed["mask"]).long()
        return dict(image=image, mask=mask)

    return data_pipe
