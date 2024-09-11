import os
import logging
import logging.config
from pathlib import Path

import jax
import optax  # https://github.com/deepmind/optax

from models.mnist_cnn import CNN

from utils import BATCH_SIZE, LEARNING_RATE, PRINT_EVERY, SEED, STEPS
from utils.arguments import parse_args
from utils.datasets import load_mnist
from utils.general import increment_path, set_logger
from utils.trainer import train


def main(args):
    curr_dir = Path(__file__).resolve().parent
    name = Path(
        args.save_path + os.sep + args.model + "_" + args.dataname + "_" + args.id
    )

    if not os.path.exists(curr_dir / name):
        os.makedirs(curr_dir / name)
        args.save_path = curr_dir / name
    else:
        args.save_path = str(
            increment_path(curr_dir / name, exist_ok=args.exist_ok, mkdir=True)
        )

    set_logger(args)
    logging.info(args)

    trainloader, testloader = load_mnist()

    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)

    if args.model == "CNN":
        model = CNN(subkey)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")

    optim = optax.adamw(LEARNING_RATE)
    model = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)


if __name__ == "__main__":
    args = parse_args()
    main(args)
