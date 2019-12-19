import numpy as np
from tensorflow.keras import backend as K


def get_single_greyscale(pixel):
    val = 232 + round(pixel * 23)
    return '\x1b[48;5;{}m \x1b[0m'.format(int(val))


def print_greyscale(pixels):
    if len(pixels.shape) == 1:
        print(''.join(get_single_greyscale(p) for p in pixels))

    elif len(pixels.shape) == 2:
        for line in pixels:
            print(''.join(get_single_greyscale(p) for p in line))

    else:
        raise ValueError(
            'Can\'t visualize tensors with more than two dimensions'
        )


def normalize_weights_to_pixels(vector, magnitude=1.0):
    if magnitude is None:
        if type(vector) is np.ndarray:
            magnitude = max(vector.max(), - vector.min())
        else:
            magnitude = max(max(vector), - min(vector))

    return vector / magnitude / 2.0 + .5


def print_vector(vector, normalize=True):
    if normalize:
        magnitude = max(1.0, vector.max(), - vector.min())
        vector = normalize_weights_to_pixels(vector, magnitude=magnitude)

    print_greyscale(vector)


def get_activations(model, layer, x_batch):
    """
    Get the model's activations at the given layer for the given batch of
    input values.
    Source: https://github.com/fchollet/keras/issues/41#issuecomment-219262860
    """
    get_activations = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer].output]
    )
    activations = get_activations([x_batch, 0])

    return activations[0]


def print_activations(model, x):
    x_batch = np.array([x])

    for i, layer in enumerate(model.layers):
        if (
            layer.name.startswith('repeat_vector') or
            layer.name.startswith('dropout')
        ):
            # Some layers add no information, so we ignore them
            continue

        activations = get_activations(model, i, [x])[0]

        print('Layer "{}":'.format(layer.name))

        print_vector(activations)
        print()
