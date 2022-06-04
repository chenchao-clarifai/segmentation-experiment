from .. import datasets


def test_base_dataset():
    dataset = datasets.SemanticSegmentationDataset("src/tests/data", "src/tests/data")
    assert len(dataset) == 1


def test_transforms():
    train_cfg = datasets.transforms.TrainTransformsConfigs()
    eval_cfg = datasets.transforms.EvalTransformsConfigs()

    datasets.transforms.get_train_transforms(train_cfg)
    datasets.transforms.get_eval_transforms(eval_cfg)

    # TODO: add some real images and test the transforms
