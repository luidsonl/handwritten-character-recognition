import numpy as np
from numpy.typing import NDArray
from emnist import extract_training_samples, extract_test_samples
from keras.utils import to_categorical


EMNIST_BALANCED_LABELS: list[str] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

NUM_CLASSES: int = len(EMNIST_BALANCED_LABELS)

DatasetTuple = tuple[
    NDArray[np.uint8], NDArray[np.uint8],
    NDArray[np.uint8], NDArray[np.uint8],
]

ProcessedTuple = tuple[
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32], NDArray[np.float32],
]


def load_emnist(split: str = "balanced") -> DatasetTuple:
    x_train, y_train = extract_training_samples(split)
    x_test, y_test = extract_test_samples(split)
    return x_train, y_train, x_test, y_test


def preprocess(
    x_train: NDArray[np.uint8],
    y_train: NDArray[np.uint8],
    x_test: NDArray[np.uint8],
    y_test: NDArray[np.uint8],
    num_classes: int = NUM_CLASSES,
) -> ProcessedTuple:
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def load_and_preprocess(split: str = "balanced") -> ProcessedTuple:
    x_train, y_train, x_test, y_test = load_emnist(split)
    return preprocess(x_train, y_train, x_test, y_test)


def get_label(index: int) -> str:
    return EMNIST_BALANCED_LABELS[index]
