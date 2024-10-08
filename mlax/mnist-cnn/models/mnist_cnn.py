import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping


class CNN(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, '1 28 28']) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    from utils import SEED
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)
    print(model)
