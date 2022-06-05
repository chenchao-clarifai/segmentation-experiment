import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Projector", "Bias", "PositiveReal"]


class Projector(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        normalized=True,
        norm_type=2.0,
        weight=None,
        channel_last=False,
    ):
        super().__init__()
        if weight is None:
            weight = torch.randn(out_dim, in_dim)
        else:
            weight = weight.cpu()
            assert weight.size() == (out_dim, in_dim)
        self.weight = nn.Parameter(weight)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalized = normalized
        self.norm_type = norm_type
        self.channel_last = channel_last

    def _normalize(self, x):
        return F.normalize(input=x, dim=-1, p=self.norm_type)

    def forward(self, inputs):

        inputs = self._normalize(inputs)
        projectors = self._normalize(self.weight)

        if self.channel_last:
            outputs = torch.einsum("...i, oi -> ...o", inputs, projectors)
        else:
            outputs = torch.einsum("bi..., oi -> bo...", inputs, projectors)

        return outputs

    def extra_repr(self):

        out = []
        out.append(f"in_dim={self.in_dim}")
        out.append(f"out_dim={self.out_dim}")
        out.append(f"normalized={self.normalized}")
        out.append(f"norm_type={self.norm_type}")

        return ", ".join(out)


class Bias(nn.Module):
    def __init__(self, num_dim: int, nonlinear: nn.Module = nn.Tanh):
        super().__init__()
        self.num_dim = num_dim
        self._bias = nn.Parameter(torch.zeros(num_dim))
        self.nonlinear = nonlinear()

    def forward(self):
        return self.nonlinear(self._bias)


class PositiveReal(nn.Module):
    def __init__(self, initial_value=1.0):
        super().__init__()
        self._log_value = nn.Parameter(torch.tensor(initial_value).log())

    def forward(self):
        return self._log_value.exp()
