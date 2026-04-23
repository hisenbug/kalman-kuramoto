import warnings
import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    warnings.warn(
        "MPS backend not available; falling back to CPU. Expect ~10x slower runs.",
        RuntimeWarning,
        stacklevel=2,
    )
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
