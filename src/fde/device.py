"""Compute device of the learners that can run on a GPU (XGBoost, CatBoost and the MLP member).

FDE runs on the CPU by default, which gives the results of the accompanying paper. With device "gpu", those learners
are trained on the first CUDA GPU; their results can differ slightly from the CPU results. Fitted models are kept on
the CPU, so that a model trained on a GPU also predicts on a machine without one.
"""
DEVICE = "cpu"
GPU_RAM_PART = 0.1                 # share of GPU memory one CatBoost model may reserve (several run at the same time)


def use(device):
    global DEVICE
    DEVICE = device


def gpu():
    return DEVICE == "gpu"


def detected():
    """Name of the first NVIDIA GPU from nvidia-smi, or None. Does not load PyTorch, so that the main process loads the
    OpenMP library of LightGBM and XGBoost first (on macOS, loading PyTorch's copy first can crash the process)."""
    import shutil, subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def available():
    """Name of the first CUDA GPU usable by PyTorch, or None (loads PyTorch)."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch
            return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None
