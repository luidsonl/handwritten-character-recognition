import os
import random

import numpy as np
import tensorflow as tf


DEFAULT_SEED = 42


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
