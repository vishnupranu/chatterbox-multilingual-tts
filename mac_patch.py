"""
mac_patch.py
Hardware compatibility helper for macOS Apple Silicon (MPS), CUDA, and CPU.
Ensures checkpoint tensors are mapped to the appropriate target device on Mac.
"""
import os
import torch

# Enable PyTorch MPS fallback for operators not natively implemented in Metal (e.g. aten::_fft_r2c)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

_PATCHED = False

def get_target_device() -> str:
    """Determine the optimal compute device."""
    env_device = os.getenv("CHATTERBOX_DEVICE")
    if env_device:
        return env_device.lower()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def apply_device_patch(device: str | None = None) -> str:
    """
    Patches torch.load to ensure checkpoints without explicit map_location
    are automatically routed to the target device (essential for Apple Silicon MPS).
    """
    global _PATCHED
    target = device or get_target_device()
    map_location = torch.device(target)

    if not _PATCHED:
        torch_load_original = torch.load

        def patched_torch_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = map_location
            return torch_load_original(*args, **kwargs)

        torch.load = patched_torch_load
        _PATCHED = True

    return target
