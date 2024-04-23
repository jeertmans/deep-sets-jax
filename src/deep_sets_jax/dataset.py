"""Deep Sets model utilies."""

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import IO, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import requests
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Shaped, UInt, jaxtyped
from tqdm import tqdm


def sample(
    key: jax.random.PRNGKey,
    *arrays: Shaped[Array, "N ..."],
    min_size: int = 3,
    max_size: int = 10,
) -> Iterator[tuple[Shaped[Array, "sample_size ..."], ...]]:
    """Generate a never-ending sequence of random samples from arrays.

    Arrays must have a leading axis with the same size.

    Each element of this sequence contains a number of samples from ``arrays``,
    where the actual number randomly chosen in ``[min_num_images, max_num_images[``.

    Args:
        key: The random key to be used.
        arrays: The sequence of arrays to sample.
        min_size: The minimum size of a sample.
        max_size: The maximum size (excluded) of a sample.

    Return:
        A sequence of random samples from ``arrays``.

    """
    array_size = arrays[0].shape[0]

    while True:
        key, key_sample_size, key_indices = jax.random.split(key, 3)
        sample_size = jax.random.randint(key_sample_size, (), min_size, max_size)
        indices = jax.random.randint(key_indices, (sample_size,), 0, array_size)
        yield tuple(jnp.take(array, indices, 0) for array in arrays)


@jaxtyped(typechecker=typechecker)
class Dataset(eqx.Module):
    """MNIST dataset class."""

    x_train: UInt[Array, "train_size 28 28"] = eqx.field(converter=jnp.asarray)
    """The train images."""
    x_test: UInt[Array, "test_size 28 28"] = eqx.field(converter=jnp.asarray)
    """The test images."""
    y_train: UInt[Array, " train_size"] = eqx.field(converter=jnp.asarray)
    """The train labels."""
    y_test: UInt[Array, " test_size"] = eqx.field(converter=jnp.asarray)
    """The test labels."""

    @cached_property
    def x_train_mean(self) -> Float[Array, " "]:
        """The mean value of the train images."""
        return self.x_train.mean()

    @cached_property
    def x_train_std(self) -> Float[Array, " "]:
        """The std value of the train images."""
        return self.x_train.std()

    @cached_property
    def normalized_x_train(self) -> Float[Array, "train_size 28 28"]:
        """The train images, normalized with respect to the train set."""
        return (self.x_train - self.x_train_mean) / self.x_train_std

    @cached_property
    def normalized_x_test(self) -> Float[Array, "test_size 28 28"]:
        """The test images, normalized with respect to the train set."""
        return (self.x_test - self.x_train_mean) / self.x_train_std

    def sample_train(
        self, *, key: jax.random.PRNGKey, **kwargs: Any
    ) -> Iterator[tuple[Float[Array, "num_images 26 26"], UInt[Array, " "]]]:
        """Sample the (normalized) train set to generate a sequence of training images.

        Each sample is a 2-tuple (images, labels) where
        the number of images is random for each sample.

        See :func:`sample` for more details.

        Args:
            key: The random key to be used.
            kwargs: Kerword arguments passed to :func:`sample`.

        Return:
            A sequence of random train samples.

        """
        for x_train, y_train in sample(
            key, self.normalized_x_train, self.y_train, **kwargs
        ):
            yield x_train, y_train.sum()

    def sample_test(
        self, *, key: jax.random.PRNGKey, **kwargs: Any
    ) -> Iterator[tuple[Float[Array, "num_images 26 26"], UInt[Array, " "]]]:
        """Sample the (normalized) test set to generate a sequence of training images.

        Each sample is a 2-tuple (images, labels) where
        the number of images is random for each sample.

        See :func:`sample` for more details.

        Args:
            key: The random key to be used.
            kwargs: Kerword arguments passed to :func:`sample`.

        Return:
            A sequence of random test samples.

        """
        for x_test, y_test in sample(
            key, self.normalized_x_test, self.y_test, **kwargs
        ):
            yield x_test, y_test.sum()

    @classmethod
    def load_archive(cls, file: str | Path | IO) -> "Dataset":
        """Load the MNIST dataset from a folder.

        Must contain four files:
        - x_train.npy
        - x_test.npy
        - y_train.npy
        - y_test.npy

        Args:
            file: The path to the file, or the file, containing the dataset.
                Must be a ``.npz`` file.

        Return:
            The dataset instance.

        """
        data = jnp.load(file)

        return cls(**data)

    @classmethod
    def download_archive(
        cls,
        url: str = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        file: str | Path = "mnist.npz",
        cached: bool = True,
        chunk_size: int = 1024,
        progress: bool = True,
    ) -> "Dataset":
        """Download (and load) the MNIST dataset from a given url.

        Args:
            url: The URL to download the dataset from.
            file: Where the archive should be stored.
            cached: If :py:data:`True`, and if ``file`` already exists,
                skip the download part. If ``file`` exists but this is set
                to :py:data:`False`, an error wll be raised.
            chunk_size: The chunk size, in bytes, used when downloading
                the data.
            progress: Whether to output a progress bar when downloading.

        Return:
            The dataset instance.

        """
        if isinstance(file, str):
            file = Path(file)

        if file.exists():
            if cached:
                return cls.load_archive(file=file)

            raise ValueError(
                f"Cannot download archive to {file} because it already exists. "
                "Please set 'cached=True' or delete the file manually."
            )

        response = requests.get(url, stream=True)
        stream = response.iter_content(chunk_size=chunk_size)
        total = int(response.headers.get("content-length", 0))

        with (
            open(file, "w+b") as f,
            tqdm(
                stream,
                desc="Downloading MNIST archive...",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=chunk_size,
                disable=not progress,
                leave=False,
            ) as bar,
        ):
            for chunk in stream:
                size = f.write(chunk)
                bar.update(size)

        return cls.load_archive(file=file)
