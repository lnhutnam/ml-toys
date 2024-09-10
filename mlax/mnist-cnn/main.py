import jax
import optax  # https://github.com/deepmind/optax

from models.mnist_cnn import CNN

from utils import BATCH_SIZE, LEARNING_RATE, PRINT_EVERY, SEED, STEPS
from utils.datasets import load_mnist
from utils.trainer import train


def main():
    trainloader, testloader = load_mnist()

    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)

    optim = optax.adamw(LEARNING_RATE)
    model = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)


if __name__ == "__main__":
    main()
