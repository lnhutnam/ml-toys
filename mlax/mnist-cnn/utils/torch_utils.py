import os
import random
import numpy as np
import torch

def init_seeds(seed: int = 0, deterministic: bool = False) -> None:
    """
    Initializes random seeds for reproducibility across various libraries and sets PyTorch
    to use deterministic algorithms if required.

    Args:
        seed (int): The seed value to use for random number generators. Defaults to 0.
        deterministic (bool): If True, sets PyTorch to use deterministic algorithms. Defaults to False.
    """
    # Set seed for random, numpy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Set cuDNN to benchmark mode for potential performance benefits
    torch.backends.cudnn.benchmark = True

    if deterministic:
        # Ensure deterministic behavior in PyTorch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

        # Set environment variables for additional determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA >= 10.2
        os.environ["PYTHONHASHSEED"] = str(seed)
