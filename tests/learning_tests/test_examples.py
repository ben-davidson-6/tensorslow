import gzip
import os
import urllib.request
from typing import Iterator, Tuple

import idx2numpy
import numpy as np
import pytest
from numpy.typing import NDArray

from tensorslow.constants import TENSORSLOW_MNIST_DIR
from tensorslow.data.datasets import Dataset
from tensorslow.gradient_tape import GradientTape
from tensorslow.layers import Linear, Relu, SoftmaxCrossEntropyLogits
from tensorslow.model import Model
from tensorslow.optimisers import SGD
from tensorslow.tensor import Tensor


@pytest.fixture
def download_mnist() -> None:
    if TENSORSLOW_MNIST_DIR.exists():
        return
    else:
        TENSORSLOW_MNIST_DIR.mkdir(exist_ok=True, parents=True)

    base = "http://yann.lecun.com/exdb/mnist"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for file in files:
        url = f"{base}/{file}"
        file_path = TENSORSLOW_MNIST_DIR.joinpath(file)
        urllib.request.urlretrieve(url, file_path)
        with gzip.open(file_path, "rb") as f:
            data = idx2numpy.convert_from_string(f.read())
            np.save(file_path.with_suffix(""), data)
            os.remove(file_path)


MnistSplit = Tuple[NDArray, NDArray]
RawMnistData = Tuple[MnistSplit, MnistSplit]


@pytest.fixture
def mnist(download_mnist: None) -> RawMnistData:
    train_images = np.load(TENSORSLOW_MNIST_DIR.joinpath("train-images-idx3-ubyte.npy"))
    train_labels = np.load(TENSORSLOW_MNIST_DIR.joinpath("train-labels-idx1-ubyte.npy"))
    test_images = np.load(TENSORSLOW_MNIST_DIR.joinpath("t10k-images-idx3-ubyte.npy"))
    test_labels = np.load(TENSORSLOW_MNIST_DIR.joinpath("t10k-labels-idx1-ubyte.npy"))
    return (train_images, train_labels), (test_images, test_labels)


@pytest.fixture
def mnist_dataset(mnist: RawMnistData) -> Tuple[Dataset, Dataset]:
    train, test = mnist

    class Data(Dataset):
        def __init__(self, mnist_split: MnistSplit) -> None:
            self.images = np.reshape(mnist_split[0], (-1, 28 * 28))
            self.images = self.images.astype(np.float32)
            self.images = (self.images - self.images.mean()) / self.images.std()

            self.labels = mnist_split[1][..., None]

            self.batch_size = 64
            self.epochs = 10

        def __iter__(self) -> Iterator[MnistSplit]:
            n_batches = self.images.shape[0] // self.batch_size
            for _ in range(self.epochs):
                for batch in range(n_batches):
                    s = batch * self.batch_size
                    e = (batch + 1) * self.batch_size
                    batch_images = self.images[s:e]
                    batch_labels = self.labels[s:e]
                    yield batch_images, batch_labels

    return Data(train), Data(test)


@pytest.mark.learning
def test_mnist_model(mnist_dataset: Tuple[Dataset, Dataset]) -> None:
    train, test = mnist_dataset

    class FC(Model):
        def __init__(self) -> None:
            super().__init__()
            self.linear_0 = Linear(128)
            self.relu = Relu()
            self.linear_1 = Linear(128)
            self.linear_2 = Linear(10)
            self.loss = SoftmaxCrossEntropyLogits()

        def forward(self, x: Tensor, labels: Tensor, loss: bool = True) -> Tensor:
            x = self.linear_0(x)
            x = self.relu(x)
            x = self.linear_1(x)
            x = self.relu(x)
            x = self.linear_2(x)
            if loss:
                x = self.loss(x, labels)
            return x

    model = FC()
    optimiser = SGD(learning_rate=0.001)
    in_tensor = Tensor(np.array(0), name="input", trainable=False)
    label_tensor = Tensor(np.array(0), name="labels", trainable=False)

    for j, (image_batch, labels) in enumerate(train):
        in_tensor.value = image_batch
        label_tensor.value = labels

        with GradientTape() as tape:
            loss = model(in_tensor, label_tensor)

        dloss_d = tape.gradients(loss)
        optimiser.minimise(dloss_d, model)

        if j % 1000 == 0:
            print(loss.value)
            print("*")

    count = 0
    correct = 0

    for j, (image_batch, labels) in enumerate(test):
        in_tensor.value = image_batch
        label_tensor.value = labels
        x = model(in_tensor, label_tensor, loss=False).value
        count += x.shape[0]
        correct += np.sum(np.argmax(x, axis=-1) == labels[:, 0])
    print(f"accuracy : {correct/count}")
