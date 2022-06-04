from .. import models


def test_projector():
    prj = models.heads.Projector(3, 3)
    print(prj)


def test_positive_real():
    t = models.heads.PositiveReal()
    print(t)


def test_bias():
    bias = models.heads.Bias(3)
    assert bias().size() == (3,)
    print(bias)
