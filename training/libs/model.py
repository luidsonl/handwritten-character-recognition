import json
import os
import struct

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.initializers import GlorotUniform, Zeros
from keras.src.models.sequential import Sequential as SequentialType

SEED = 42


def build_model(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 47,
) -> SequentialType:
    model = Sequential([
        Conv2D(32, (5, 5), padding='same', activation='tanh', input_shape=input_shape,
               kernel_initializer=GlorotUniform(seed=SEED), bias_initializer=Zeros()),
        MaxPooling2D(pool_size=2, strides=2),

        Conv2D(48, (5, 5), padding='same', activation='tanh',
               kernel_initializer=GlorotUniform(seed=SEED), bias_initializer=Zeros()),
        MaxPooling2D(pool_size=2, strides=2),

        Conv2D(64, (5, 5), padding='same', activation='tanh',
               kernel_initializer=GlorotUniform(seed=SEED), bias_initializer=Zeros()),

        Flatten(),

        Dense(512, activation='tanh',
              kernel_initializer=GlorotUniform(seed=SEED), bias_initializer=Zeros()),
        Dense(84, activation='tanh',
              kernel_initializer=GlorotUniform(seed=SEED), bias_initializer=Zeros()),
        Dense(num_classes, activation='softmax',
              kernel_initializer=GlorotUniform(seed=SEED), bias_initializer=Zeros()),
    ])
    return model


def _build_tfjs_topology(model: SequentialType) -> dict:
    layers = []
    for layer in model.layers:
        config = layer.get_config()
        layer_type = type(layer).__name__
        tfjs_config = {}

        if layer_type == 'Conv2D':
            tfjs_config = {
                'filters': config['filters'],
                'kernel_size': list(config['kernel_size']),
                'strides': list(config['strides']),
                'padding': config['padding'].upper(),
                'data_format': 'channels_last',
                'dilation_rate': list(config.get('dilation_rate', (1, 1))),
                'groups': config.get('groups', 1),
                'activation': config['activation'] if isinstance(config['activation'], str) else config['activation'].__name__,
                'use_bias': config['use_bias'],
                'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}},
                'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            }

        elif layer_type == 'MaxPooling2D':
            tfjs_config = {
                'pool_size': list(config['pool_size']),
                'strides': list(config['strides']),
                'padding': config['padding'].upper(),
                'data_format': 'channels_last',
            }

        elif layer_type == 'Flatten':
            tfjs_config = {'data_format': 'channels_last'}

        elif layer_type == 'Dense':
            tfjs_config = {
                'units': config['units'],
                'activation': config['activation'] if isinstance(config['activation'], str) else config['activation'].__name__,
                'use_bias': config['use_bias'],
                'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}},
                'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            }

        layers.append({'class_name': layer_type, 'config': tfjs_config})

    return {
        'class_name': 'Sequential',
        'config': {'name': 'sequential', 'layers': layers},
    }


def _get_weight_names(model: SequentialType) -> list[tuple[str, tuple[int, ...]]]:
    names = []
    for layer in model.layers:
        layer_type = type(layer).__name__
        weights = layer.get_weights()
        if layer_type == 'Conv2D':
            names.append((f'{layer.name}/kernel', weights[0].shape))
            if len(weights) > 1:
                names.append((f'{layer.name}/bias', weights[1].shape))
        elif layer_type == 'Dense':
            names.append((f'{layer.name}/kernel', weights[0].shape))
            if len(weights) > 1:
                names.append((f'{layer.name}/bias', weights[1].shape))
    return names


def convert_to_tfjs(model: SequentialType, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    topology = _build_tfjs_topology(model)
    weight_specs = _get_weight_names(model)

    all_weights = []
    for layer in model.layers:
        for w in layer.get_weights():
            all_weights.append(w.astype(np.float32).flatten())

    weights_data = np.concatenate(all_weights).tobytes()
    shard_name = 'group1-shard1of1.bin'
    with open(os.path.join(output_dir, shard_name), 'wb') as f:
        f.write(weights_data)

    weights_manifest = [[{
        'name': name,
        'shape': list(shape),
        'dtype': 'float32',
    } for name, shape in weight_specs]]

    model_json = {
        'format': 'layers-model',
        'generatedBy': 'custom-converter',
        'convertedBy': None,
        'modelTopology': topology,
        'weightsManifest': weights_manifest,
    }

    with open(os.path.join(output_dir, 'model.json'), 'w') as f:
        json.dump(model_json, f)
