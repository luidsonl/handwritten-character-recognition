import gzip
import struct
import zipfile
from pathlib import Path

import numpy as np
import requests
from numpy.typing import NDArray
from keras.utils import to_categorical


DATA_DIR = Path(__file__).parent.parent / "data"
EMNIST_URLS = [
    "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
    "https://www.nist.gov/system/files/documents/2017/08/23/gzip.zip",
]

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


def _read_idx_images(filepath: Path) -> NDArray[np.uint8]:
    with gzip.open(filepath, "rb") as f:
        _magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols)


def _read_idx_labels(filepath: Path) -> NDArray[np.uint8]:
    with gzip.open(filepath, "rb") as f:
        _magic, num_labels = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def _download_emnist() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR / "emnist.zip"
    extracted_marker = DATA_DIR / ".extraction_done"

    if extracted_marker.exists():
        return

    if not zip_path.exists():
        print("Downloading EMNIST balanced dataset...")
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        for url in EMNIST_URLS:
            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print("Download complete.")
                break
            except Exception as e:
                print(f"  Failed: {url} ({e})")
                continue
        else:
            raise RuntimeError("All download URLs failed")
    else:
        print("Using existing EMNIST zip file.")

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if "balanced" in name.lower():
                zf.extract(name, DATA_DIR)
    print("Extraction complete.")

    extracted_marker.touch()


def _get_balanced_paths() -> dict[str, Path]:
    base = DATA_DIR / "gzip"
    return {
        "x_train": base / "emnist-balanced-train-images-idx3-ubyte.gz",
        "y_train": base / "emnist-balanced-train-labels-idx1-ubyte.gz",
        "x_test": base / "emnist-balanced-test-images-idx3-ubyte.gz",
        "y_test": base / "emnist-balanced-test-labels-idx1-ubyte.gz",
    }


def load_emnist(split: str = "balanced") -> DatasetTuple:
    _download_emnist()
    paths = _get_balanced_paths()

    x_train = _read_idx_images(paths["x_train"])
    y_train = _read_idx_labels(paths["y_train"])
    x_test = _read_idx_images(paths["x_test"])
    y_test = _read_idx_labels(paths["y_test"])

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
