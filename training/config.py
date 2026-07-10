# Training Configuration
# =====================

# Dataset
DATASET_SPLIT = "balanced"
NUM_CLASSES = 47
IMAGE_SHAPE = (28, 28, 1)

# CNN Architecture
ARCHITECTURE = {
    "conv_block_1": {
        "filters": 32,
        "kernel_size": (5, 5),
        "padding": "same",
        "activation": "tanh",
    },
    "pool_block_1": {
        "pool_size": 2,
        "strides": 2,
    },
    "conv_block_2": {
        "filters": 48,
        "kernel_size": (5, 5),
        "padding": "same",
        "activation": "tanh",
    },
    "pool_block_2": {
        "pool_size": 2,
        "strides": 2,
    },
    "conv_block_3": {
        "filters": 64,
        "kernel_size": (5, 5),
        "padding": "same",
        "activation": "tanh",
    },
    "dense_1": {
        "units": 512,
        "activation": "tanh",
    },
    "dense_2": {
        "units": 84,
        "activation": "tanh",
    },
    "output": {
        "units": 47,
        "activation": "softmax",
    },
}

# Compilation
LOSS_FUNCTION = "categorical_crossentropy"
OPTIMIZER = "adam"
LEARNING_RATE = 0.001
METRICS = ["accuracy"]

# Training
EPOCHS = 15
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 5

# Labels mapping (47 classes - EMNIST Balanced)
LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't',
]
