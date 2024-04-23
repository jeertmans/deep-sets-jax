"""Deep Sets model utilies."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, UInt


class DeepSets(eqx.Module):
    """A (very basic) Deep Sets model."""

    layers: list
    """The list of layers."""

    def __init__(self, key: jax.random.PRNGKey):
        """
        Initialize a Deep Sets model.

        Args:
            key: The random key to be used.
        """
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        self.layers = [
            lambda x: jnp.expand_dims(
                x, axis=1
            ),  # Turn input to [num_images, 1, 28, 28]
            # We need to use vmap because JAX does not automatically map over leading axis
            jax.vmap(eqx.nn.Conv2d(1, 10, kernel_size=5, key=key1)),
            jax.vmap(eqx.nn.MaxPool2d(kernel_size=2)),
            jax.nn.relu,
            jax.vmap(eqx.nn.Conv2d(10, 20, kernel_size=5, key=key2)),
            jax.vmap(eqx.nn.MaxPool2d(kernel_size=2)),
            jax.nn.relu,
            # We sum over `num_images`
            lambda x: jnp.sum(x, axis=0, keepdims=True),
            jnp.ravel,
            eqx.nn.Linear(6480, 500, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(500, 10, key=key4),
            jax.nn.relu,
            eqx.nn.Linear(10, 1, key=key5),
        ]

    def __call__(self, x: Float[Array, "num_images 28 28"]) -> Float[Array, " "]:  # noqa: D102
        for layer in self.layers:
            x = layer(x)

        return x.sum()


@eqx.filter_jit
def loss(
    model: DeepSets,
    x: Float[Array, "num_images 28 28"],
    y_true: UInt[Array, " "],
    plot: bool = False,
) -> Float[Array, " "]:
    """
    Compute the loss of the model with respect to a ground truth output.

    Args:
        model: The DeepSets model.
        x: The train or test input images.
        y_true: The ground truth, i.e., the expected sum.
        plot: Whether to plot the images.
    """
    y_pred = model(x)

    if plot:
        import math

        import matplotlib.pyplot as plt

        num_images = x.shape[0]

        nrows = math.ceil(math.sqrt(num_images))
        ncols = math.ceil(num_images / nrows)

        fig, axes = plt.subplots(nrows, ncols)
        fig.suptitle(f"Model predicted {y_pred:.1f} - True value is {y_true:.1f}")

        for index, image in enumerate(x):
            i, j = divmod(index, ncols)
            axes[i, j].pcolormesh(image[::-1, ::+1])
            axes[i, j].axis("off")
    return (y_true - y_pred).sum() ** 2
