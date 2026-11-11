import importlib.metadata
import platform
import sys
from pathlib import Path

import distro
from torch.cuda import is_available as torch_cuda_available
lazy import torch

try:
    from lineremovernn._lineremovernn_ext import (  # noqa: F401
        Dataset,
        generate_pages,
    )

    HAS_EXT = True
except ImportError:
    HAS_EXT = False


def _probe_autocast(device: torch.device) -> bool:
    """Check whether fp16 autocast is numerically safe on this device.

    Probes with the real model when available, since the fp16 failures on
    some GPU/driver/cuDNN combos (notably GTX 16xx) are kernel-specific:
    a small synthetic network can pass while the real one produces NaNs.
    """
    if device.type != "cuda":
        return False

    try:
        torch.manual_seed(0)
        try:
            from lineremovernn.model.lineremover import LineRemovalUNet

            model = LineRemovalUNet().to(device).train()
            # Realistic inference shapes, plus a stressed one (>1 values).
            test_inputs = [
                torch.rand(1, 1, 256, 256, device=device),
                torch.rand(2, 1, 384, 384, device=device) * 4.0,
            ]
        except Exception:  # noqa: BLE001
            model = (
                torch.nn.Sequential(
                    torch.nn.Conv2d(1, 16, 3, padding=1, bias=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(16, 16, 3, padding=1, bias=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(16, 1, 3, padding=1, bias=False),
                )
                .to(device)
                .train()
            )
            test_inputs = [torch.rand(4, 1, 64, 64, device=device) * 4.0]

        with torch.no_grad():
            for x in test_inputs:
                reference = model(x).float()
                with torch.autocast(device.type, dtype=torch.float16):
                    out = model(x).float()

                if not torch.isfinite(out).all():
                    return False
                scale = reference.abs().max().clamp_min(1e-6)
                if ((out - reference).abs().max() / scale).item() > 0.05:
                    return False
        return True
    except Exception:  # noqa: BLE001
        print("AMP probe failed with an exception; disabling autocast.")
        return False


ROOT = Path(__file__).parent.parent.parent  # equal src dir
SAVED = ROOT.parent / "saved"

DEFAULT_DOWNLOADS = SAVED / "downloads"
DEFAULT_DATASETS = SAVED / "datasets"
DEFAULT_MODELS = SAVED / "models"
DEFAULT_LOGS = SAVED / "logs"

VERSION = importlib.metadata.version("lineremovernn")
PYTHON_VERSION = ".".join(map(str, sys.version_info[:3]))
if platform.system() == "Windows":
    OS = f"Windows {platform.release()}"
elif platform.system() == "Darwin":
    OS = f"macOS {platform.mac_ver()[0]}"
elif platform.system() == "Linux":
    OS = distro.name(pretty=True)
ARCH = " ".join(platform.architecture())
DEVICE = "cuda" if torch_cuda_available() else "cpu"
USE_AMP: bool = _probe_autocast(torch.device(DEVICE))
