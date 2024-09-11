import torch
import torchvision
from utils import BATCH_SIZE

def load_mnist(data_dir: str = "MNIST", batch_size: int = BATCH_SIZE):
    """
    Loads the MNIST dataset and returns DataLoader objects for training and testing.

    Args:
        data_dir (str): The directory to store/download the MNIST dataset. Defaults to "MNIST".
        batch_size (int): The batch size for the DataLoader. Defaults to BATCH_SIZE.

    Returns:
        tuple: A tuple containing the trainloader and testloader.
    """
    # Define the transform to normalize the data
    normalize_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    
    # Load the MNIST training and testing datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=normalize_data,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=normalize_data,
    )
    
    # Create DataLoaders for the training and testing datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False  # Shuffle set to False for testing
    )

    return train_loader, test_loader
