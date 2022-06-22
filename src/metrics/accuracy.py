import torch

from .base import MetricBase


class Accuracy(MetricBase):

    EPS = 1e-8

    def __init__(self) -> None:
        self.per_item_accuracy = []
        self._sum_of_accuracy = 0.0
        self._counts = 0

    def evaluate(self, predict: torch.LongTensor, target: torch.LongTensor) -> float:
        acc = (predict == target).float().mean()
        return acc.item()

    def accumulate(self, acc: float) -> None:
        self.per_item_accuracy.append(acc)
        self._sum_of_accuracy += acc
        self._counts += acc

    def results(self):
        return self._sum_of_accuracy / max(self._counts, self.EPS)
