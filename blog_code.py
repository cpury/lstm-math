"""
blog_code.py
The code from my blog post at http://cpury.github.io/learning-math/ .
This is a simplified version of the other code we have here.
"""

import itertools
import random

def generate_equations(shuffle=True, max_count=None):
    """
    Generates all possible math equations given the global configuration.
    If max_count is given, returns that many at most. If shuffle is True,
    the equation will be generated in random order.
    """
    # Generate all possible unique sets of numbers
    number_permutations = itertools.permutations(
        range(MIN_NUMBER, MAX_NUMBER + 1), 2
    )

    # Shuffle if required. The downside is we need to convert to list first
    if shuffle:
        number_permutations = list(number_permutations)
        random.shuffle(number_permutations)

    # If a max_count is given, use itertools to only look at that many items
    if max_count is not None:
        number_permutations = itertools.islice(number_permutations, max_count)

    # Build an equation string for each and yield to caller
    for x, y in number_permutations:
        yield '{} + {}'.format(x, y)

import numpy as np

CHARS = [' ', '+', '\0'] + [str(n) for n in range(10)]
CHAR_TO_INDEX = {i: c for c, i in enumerate(CHARS)}
INDEX_TO_CHAR = {c: i for c, i in enumerate(CHARS)}

def one_hot_to_index(vector):
    if not np.any(vector):
        return -1

    return np.argmax(vector)

def one_hot_to_char(vector):
    index = one_hot_to_index(vector)
    if index == -1:
        return ''

    return INDEX_TO_CHAR[index]

def one_hot_to_string(matrix):
    return ''.join(one_hot_to_char(vector) for vector in matrix)

def equations_to_x_y(equations, n):
    """
    Given a list of equations, converts them to one-hot vectors to build
    two data matrixes x and y.
    """
    x = np.zeros((n, MAX_EQUATION_LENGTH, N_FEATURES), dtype=np.float32)
    y = np.zeros((n, MAX_RESULT_LENGTH, N_FEATURES), dtype=np.float32)

    # Get the first n_test equations and convert to test vectors
    for i, equation in enumerate(itertools.islice(equations, n)):
        result = str(eval(equation))

        # Pad the result with spaces
        result = ' ' * (MAX_RESULT_LENGTH - 1 - len(result)) + result

        # We end each sequence in a sequence-end-character:
        equation += '\0'
        result += '\0'

        for t, char in enumerate(equation):
            x[i, t, CHAR_TO_INDEX[char]] = 1

        for t, char in enumerate(result):
            y[i, t, CHAR_TO_INDEX[char]] = 1

    return x, y


def build_dataset():
    """
    Generates equations based on global config, splits them into train and test
    sets, and returns (x_test, y_test, x_train, y_train).
    """
    generator = generate_equations(max_count=N_EXAMPLES)

    # Split into training and test set based on SPLIT:
    n_test = round(SPLIT * N_EXAMPLES)
    n_train = N_EXAMPLES - n_test

    x_test, y_test = equations_to_x_y(generator, n_test)
    x_train, y_train = equations_to_x_y(generator, n_train)

    return x_test, y_test, x_train, y_train


def print_example_predictions(count, model, x_test, y_test):
    """
    Print some example predictions along with their target from the test set.
    """
    print('Examples:')

    # Pick some random indices from the test set
    prediction_indices = np.random.choice(
        x_test.shape[0], size=count, replace=False
    )
    # Get a prediction of each
    predictions = model.predict(x_test[prediction_indices, :])

    for i in range(count):
        print('{} = {}   (expected: {})'.format(
            one_hot_to_string(x_test[prediction_indices[i]]),
            one_hot_to_string(predictions[i]),
            one_hot_to_string(y_test[prediction_indices[i]]),
        ))

from tensorflow import keras

def build_model():
    """
    Builds and returns the model based on the global config.
    """
    input_shape = (MAX_EQUATION_LENGTH, N_FEATURES)

    model = keras.Sequential()

    # Encoder:
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(20), input_shape=input_shape))

    # The RepeatVector-layer repeats the input n times
    model.add(keras.layers.RepeatVector(MAX_RESULT_LENGTH))

    # Decoder:
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(20, return_sequences=True)))

    model.add(keras.layers.TimeDistributed(keras.layers.Dense(N_FEATURES)))
    model.add(keras.layers.Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(lr=0.01),
        metrics=['accuracy'],
    )

    return model

def main():
    # Fix the random seed to get a consistent dataset
    random.seed(RANDOM_SEED)

    x_test, y_test, x_train, y_train = build_dataset()

    model = build_model()

    model.summary()
    print()

    # Let's print some predictions now to get a feeling for the equations
    print()
    print_example_predictions(5, model, x_test, y_test)
    print()

    try:
        model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    'model.h5',
                    save_best_only=True,
                ),
            ]
        )
    except KeyboardInterrupt:
        print('\nCaught SIGINT\n')

    # Load weights achieving best val_loss from training:
    model.load_weights('model.h5')

    print_example_predictions(20, model, x_test, y_test)


def predict(model, equation):
    """
    Given a model and an equation string, returns the predicted result.
    """
    x = np.zeros((1, MAX_EQUATION_LENGTH, N_FEATURES), dtype=np.bool)
    equation += '\0'

    for t, char in enumerate(equation):
        x[0, t, CHAR_TO_INDEX[char]] = 1

    predictions = model.predict(x)
    return one_hot_to_string(predictions[0])[:-1]


MIN_NUMBER = 0
MAX_NUMBER = 999

MAX_N_EXAMPLES = (MAX_NUMBER - MIN_NUMBER) ** 2
N_EXAMPLES = 30000
N_FEATURES = len(CHARS)
MAX_NUMBER_LENGTH_LEFT_SIDE = len(str(MAX_NUMBER))
MAX_NUMBER_LENGTH_RIGHT_SIDE = MAX_NUMBER_LENGTH_LEFT_SIDE + 1
MAX_EQUATION_LENGTH = (MAX_NUMBER_LENGTH_LEFT_SIDE * 2) + 4
MAX_RESULT_LENGTH = MAX_NUMBER_LENGTH_RIGHT_SIDE + 1

SPLIT = .1
EPOCHS = 200
BATCH_SIZE = 256

RANDOM_SEED = 1


if __name__ == '__main__':
    main()
