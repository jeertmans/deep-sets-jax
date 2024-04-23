"""Deep Sets model utilies."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, UInt


class Phi(eqx.Module):
    """
    Phi function from the Deep Sets structure.

    This function is applied to each image separately.
    """

    layers: list
    """The list of layers."""

    def __init__(self, *, key: jax.random.PRNGKey):
        """
        Initialize the model.

        Args:
            key: The random key to be used.
        """
        key1, key2 = jax.random.split(key, 2)
        self.layers = [
            eqx.nn.Conv2d(1, 10, kernel_size=5, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            eqx.nn.Conv2d(10, 20, kernel_size=5, key=key2),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, " "]:
        """
        Evaluate the model.

        Arg:
            x: The input image.

        Return:
            The output of the model.
        """
        for layer in self.layers:
            x = layer(x)

        return x


class Rho(eqx.Module):
    """
    Rho function from the Deep Sets structure.

    This function is applied to all the images combined,
    after they have been passed in Phi.
    """

    layers: list
    """The list of layers."""

    def __init__(self, *, key: jax.random.PRNGKey):
        """
        Initialize the model.

        Args:
            key: The random key to be used.
        """
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(6480, 500, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(500, 50, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(50, 10, key=key3),
            jax.nn.relu,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, " "]:
        """
        Evaluate the model.

        Arg:
            x: The (intermediate) input.

        Return:
            The output of the model.
        """
        for layer in self.layers:
            x = layer(x)

        x = x * jnp.arange(10)  # Each of the 10 outputs is assigned a digit

        return x.sum()


class DeepSets(eqx.Module):
    """A (very basic) Deep Sets model."""

    phi: Phi
    rho: Rho

    def __init__(self, *, key: jax.random.PRNGKey):
        """
        Initialize a Deep Sets model.

        Args:
            key: The random key to be used.
        """
        key1, key2 = jax.random.split(key, 2)

        self.phi = Phi(key=key1)
        self.rho = Rho(key=key2)

    def __call__(self, x: Float[Array, "num_images 28 28"]) -> Float[Array, " "]:
        """
        Evaluate the model.

        Arg:
            x: The input images.

        Return:
            The output of the model.
        """
        x = jnp.expand_dims(x, axis=1)
        x = jax.vmap(self.phi)(x)
        # We sum over `num_images`
        x = jnp.sum(x, axis=0)
        x = jnp.ravel(x)
        x = self.rho(x)

        return x


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
