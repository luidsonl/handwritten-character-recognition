from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.models.sequential import Sequential as SequentialType


def build_model(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 47,
) -> SequentialType:
    model = Sequential([
        Conv2D(32, (5, 5), padding='same', activation='tanh', input_shape=input_shape),
        MaxPooling2D(pool_size=2, strides=2),

        Conv2D(48, (5, 5), padding='same', activation='tanh'),
        MaxPooling2D(pool_size=2, strides=2),

        Conv2D(64, (5, 5), padding='same', activation='tanh'),

        Flatten(),

        Dense(512, activation='tanh'),
        Dense(84, activation='tanh'),
        Dense(num_classes, activation='softmax'),
    ])
    return model
