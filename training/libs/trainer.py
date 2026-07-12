import os
from numpy.typing import NDArray
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.models.sequential import Sequential as SequentialType
from keras.src.callbacks.history import History


def compile_model(
    model: SequentialType,
    learning_rate: float = 0.001,
) -> SequentialType:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_model(
    model: SequentialType,
    x_train: NDArray,
    y_train: NDArray,
    epochs: int = 20,
    batch_size: int = 128,
    validation_split: float = 0.1,
    checkpoint_path: str | None = None,
    patience: int = 5,
) -> History:
    callbacks: list = []

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
            )
        )

    callbacks.append(
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def save_model_keras(model: SequentialType, path: str = "emnist_model.keras") -> str:
    model.save(path)
    return path
