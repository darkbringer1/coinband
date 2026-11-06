"""Run Timestral's transformer on an Apple-silicon GPU (MPS) while the rest stays on the CPU.

timestral keeps its input scaling statistics in float64 ("accumulated exactly as in training"), which
MPS cannot hold, so `from_pretrained(..., device="mps")` fails. The float64 work is cheap; the transformer
is the expensive part and runs in float32. `load_forecaster(..., "mps")` loads the model on the CPU and
moves only `model.transformer` to the GPU behind a wrapper that ships its inputs over and its output back.
Forecasts match the CPU to float32 rounding (about 2e-7 relative) and run about 3x faster for 209M.

This wraps upstream objects at runtime; no timestral code is copied or changed.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn


def _to(x, device):
    """Move tensors inside tensors, dataclasses, named tuples, lists, tuples and dicts to `device`."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        fields = {f.name: _to(getattr(x, f.name), device) for f in dataclasses.fields(x) if f.init}
        return dataclasses.replace(x, **fields)
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*(_to(v, device) for v in x))
    if isinstance(x, (list, tuple)):
        return type(x)(_to(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    return x


class Offload(nn.Module):
    """Runs `inner` on `device`; inputs are moved there and the output comes back to the CPU."""

    def __init__(self, inner: nn.Module, device: str) -> None:
        super().__init__()
        self.inner = inner.to(device)
        self.device = device

    def forward(self, *args, **kwargs):
        return self.inner(*_to(args, self.device), **_to(kwargs, self.device)).to("cpu")


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def load_forecaster(model_id: str, device: str = "cpu"):
    """`TimestralForecaster` for `model_id` on `device` ("cpu", "cuda" or "mps", the last via `Offload`)."""
    from timestral import TimestralForecaster

    if device != "mps":
        return TimestralForecaster.from_pretrained(model_id, device=device)
    forecaster = TimestralForecaster.from_pretrained(model_id, device="cpu")
    forecaster.model.transformer = Offload(forecaster.model.transformer, "mps")
    return forecaster
