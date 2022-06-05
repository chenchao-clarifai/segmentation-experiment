import torch

from .. import models


def test_projector():
    prj = models.heads.Projector(3, 3)
    scripted = torch.jit.script(prj)
    print(scripted)


def test_projector_channel_first():
    prj = models.heads.Projector(10, 5, channel_last=False)
    x = torch.randn((2, 10, 3, 3, 3))
    prj = torch.jit.script(prj)
    y = prj(x)
    assert y.size() == (2, 5, 3, 3, 3)


def test_projector_channel_last():
    prj = models.heads.Projector(10, 5, channel_last=True)
    x = torch.randn((2, 3, 3, 3, 10))
    prj = torch.jit.script(prj)
    y = prj(x)
    assert y.size() == (2, 3, 3, 3, 5)


def test_positive_real():
    t = models.heads.PositiveReal()
    scripted = torch.jit.script(t)
    print(scripted)


def test_bias():
    bias = models.heads.Bias(3)
    scripted = torch.jit.script(bias)
    print(scripted)
    assert bias().size() == (3,)
    print(bias)
