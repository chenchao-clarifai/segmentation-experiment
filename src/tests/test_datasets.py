from .. import datasets


def test_base_dataset():
    dataset = datasets.SemanticSegmentationDataset("src/tests/data", "src/tests/data")
    assert len(dataset) == 1
