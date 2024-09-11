import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

import torch
import torch.utils

from models.mnist_cnn import CNN
from utils.loss import loss


@eqx.filter_jit
def compute_accuracy(
    model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """
    Computes the average accuracy of the model on a given batch.

    Args:
        model (CNN): The neural network model.
        x (Float[Array, "batch 1 28 28"]): The input data batch.
        y (Int[Array, "batch"]): The true labels for the batch.

    Returns:
        Float[Array, ""]: The average accuracy for the batch.
    """
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(model: CNN, testloader: torch.utils.data.DataLoader) -> tuple[float, float]:
    """
    Evaluates the model on the test dataset, computing both the average loss and the average accuracy.

    Args:
        model (CNN): The neural network model.
        testloader (DataLoader): A PyTorch DataLoader containing the test data.

    Returns:
        tuple[float, float]: The average loss and average accuracy over the test dataset.
    """
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(testloader), avg_acc / len(testloader)
