import torch
import torchvision

from utils import BATCH_SIZE


def load_mnist():
    normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        "MNIST",
        train=True,
        download=True,
        transform=normalise_data,
    )
    test_dataset = torchvision.datasets.MNIST(
        "MNIST",
        train=False,
        download=True,
        transform=normalise_data,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    return trainloader, testloader
