import numpy as np
from PIL import Image

from .. import datasets


def test_base_dataset():
    dataset = datasets.SemanticSegmentationDataset("src/tests/data", "src/tests/data")
    assert len(dataset) == 1


def test_transforms():
    train_cfg = datasets.transforms.TrainTransformsConfigs()
    eval_cfg = datasets.transforms.EvalTransformsConfigs()

    train_transforms = datasets.transforms.get_train_transforms(train_cfg)
    eval_transforms = datasets.transforms.get_eval_transforms(eval_cfg)

    image = Image.fromarray(
        (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    ).convert("RGB")
    mask = Image.fromarray((np.random.rand(100, 100) * 255).astype(np.uint8)).convert(
        "L"
    )

    train_transformed = train_transforms(image=image, mask=mask)
    eval_transformed = eval_transforms(image=image, mask=mask)

    assert train_transformed["image"].size() == (
        3,
        train_cfg.image_size,
        train_cfg.image_size,
    )
    assert eval_transformed["image"].size() == (
        3,
        train_cfg.image_size,
        train_cfg.image_size,
    )
    assert train_transformed["mask"].size() == (
        train_cfg.image_size,
        train_cfg.image_size,
    )
    assert eval_transformed["mask"].size() == (
        train_cfg.image_size,
        train_cfg.image_size,
    )
