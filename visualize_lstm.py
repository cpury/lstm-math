import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed

from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed


def _copy_weights(trained_model, new_model):
    """
    Given a old, trained model, and a new, random model, copy the trained
    weights over to the new one.
    """
    for i, new_layer in enumerate(new_model.layers):
        if not new_layer._trainable_weights:
            continue

        trained_layer = trained_model.layers[i]

        print('Loading layer', new_layer.name, 'from', trained_layer.name)
        new_layer.set_weights(trained_layer.get_weights())


def build_stateful_model_with_weights(trained_model, length=None):
    """
    Builds a model similar to the standard blog post model, but makes it
    stateful. Pass a blogpost model in to load its weights. Set length to
    1 if you want to go through the model layer by layer.
    """
    if length is None:
        length = MAX_EQUATION_LENGTH

    batch_input_shape = (1, length, N_FEATURES)

    model = Sequential()

    # Encoder:
    model.add(LSTM(
        256, batch_input_shape=batch_input_shape, stateful=True,
    ))
    model.add(Dropout(0.25))

    # The RepeatVector-layer repeats the input n times
    model.add(RepeatVector(MAX_RESULT_LENGTH))

    # Decoder:
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))

    model.add(TimeDistributed(Dense(N_FEATURES)))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    _copy_weights(trained_model, model)

    return model


def get_lstm_output(layer, x):
    """
    Given a layer and an input, calculates the output.
    """
    return K.function(
        [layer.input],
        [layer.output]
    )([x])[0]


def get_activations_char_by_char_old(
    model, input_string, layer_i=0, stateful_model=None
):
    """
    Given a model and an input_string, returns the activation of the first
    layer after each character.

    This is an older implementation using stateful LSTMs.

    If you already have a stateful_model with length=1, you can pass it in.
    """
    stateful_model = (
        stateful_model or build_stateful_model_with_weights(model, 1)
    )
    layer = stateful_model.layers[layer_i]
    stateful_model.reset_states()

    current_input_string = ''
    activations = [None] * len(input_string)

    for i, char in enumerate(input_string):
        current_input_string += char

        x = np.zeros(stateful_model.layers[0].batch_input_shape)
        x[0, 0, CHAR_TO_INDEX[char]] = 1

        output = get_lstm_output(layer, x)

        activations[i] = get_lstm_output(layer, x)

    return activations


def get_activations_char_by_char(
    model, input_string, layer_i=0
):
    """
    Given a model and an input_string, returns the activation of the first
    layer after each character.
    """
    layer = model.layers[layer_i]

    current_input_string = ''
    activations = np.zeros((len(input_string), layer.units))

    for i, char in enumerate(input_string):
        current_input_string += char

        x = np.zeros((1,) + model.input_shape[1:])
        x[0, i, CHAR_TO_INDEX[char]] = 1

        output = get_lstm_output(layer, x)

        activations[i] = get_lstm_output(layer, x)[0]

    return activations


def plot_weights(weights, labels=None):
    """
    Given a matrix of weights and a list of labels, plots them in a heatmap.
    If labels is a list of lists, will use each for a row in the plot.
    """
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(weights, vmin=-1., vmax=1., cmap='bwr', interpolation='nearest')

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    def annotate(label, x, y):
        ax.annotate(label, (x, y), va='center', ha='center')

    if labels and type(labels[0]) is list:
        for label, y in zip(labels, range(weights.shape[0])):
            for x, char_label in enumerate(label or []):
                annotate(char_label, x, y)
    else:
        for x, label in enumerate(labels or []):
            for y in range(weights.shape[0]):
                annotate(label, x, y)

    plt.show()


def plot_activations(
    model, input_string, layer_i=0, weight_i=None,
):
    """
    Given a model and an input_string, plots the activations of each neuron
    in the first layer for each char.
    """
    if weight_i is None:
        weight_i = range(model.layers[layer_i].units)

    if type(weight_i) is int:
        weight_i = [weight_i]

    activations = get_activations_char_by_char(
        model, input_string, layer_i=layer_i,
    )

    weights = np.zeros((len(weight_i), len(input_string)))
    labels = list(input_string.replace('\0', '\\0'))

    for i, activation in enumerate(activations):
        for j, wi in enumerate(weight_i):
            weights[j, i] = activation[wi]

    plot_weights(weights, labels)


def plot_activations_single_weights(
    model, input_strings, weight_i, layer_i=0,
):
    """
    Given a model, a list of input_strings and a weight index, plots the
    activations of that neuron for each of the input strings.
    """
    # TODO Finish this

    if type(input_strings) is str:
        input_strings = [input_strings]

    max_len = max(len(input_string) for input_string in input_strings)

    weights = np.zeros((len(input_strings), max_len))
    labels = [
        list(input_string.replace('\0', '\\0'))
        for input_string in input_strings
    ]

    for i, input_string in enumerate(input_strings):
        activations = get_activations_char_by_char(
            model, input_string, layer_i=layer_i,
        )

        weights[i] = activations[:, weight_i]

    plot_weights(weights, labels)
