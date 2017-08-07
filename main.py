"""
lstm-math/main.py
Train a Seq2Seq network to predict the result of math equations on the
character level.
Configuration through global variables because I'm lazy.

Written by Max Schumacher (@cpury) in Summer 2017.
"""


from __future__ import print_function
import random
import itertools
from time import sleep

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, RepeatVector
from keras.layers.wrappers import TimeDistributed

from encode import OneHotEncoder
from visualize import print_activations


MIN_NUMBER = 0
MAX_NUMBER = 999
DECIMALS = 0

OPERATIONS = ['+']
N_OPERATIONS = 1

MAX_N_EXAMPLES = (
    (MAX_NUMBER - MIN_NUMBER) ** (N_OPERATIONS + 1) * len(OPERATIONS)
)
N_EXAMPLES = int(round(MAX_N_EXAMPLES / 2.))
N_FEATURES = 10 + len(OPERATIONS) + 2
MAX_NUMBER_LENGTH_LEFT_SIDE = (
    max(len(str(MAX_NUMBER)), len(str(MIN_NUMBER))) + DECIMALS + 1
)
MAX_NUMBER_LENGTH_RIGHT_SIDE = (
    MAX_NUMBER_LENGTH_LEFT_SIDE * (N_OPERATIONS + 1) + 1 + DECIMALS + 1
)
MAX_EQUATION_LENGTH = (MAX_NUMBER_LENGTH_LEFT_SIDE + 2) * (1 + N_OPERATIONS)
MAX_RESULT_LENGTH = MAX_NUMBER_LENGTH_RIGHT_SIDE

SPLIT = .5
EPOCHS = 800
BATCH_SIZE = 128
HIDDEN_SIZE = 32
ENCODER_DEPTH = 1
DECODER_DEPTH = 1
DROPOUT = 0

encoder = OneHotEncoder(OPERATIONS)


def prediction_to_string(matrix):
    """
    Given the output matrix of the neural network, takes the most likely
    char predicted at each point and returns the whole string.
    """
    return ''.join(
        encoder.one_hot_index_to_char(np.argmax(vector))
        for vector in matrix
    )


def to_padded_string(number, padding=None, decimals=None):
    """
    Given a number object, converts that to a string. For non-natural numbers,
    we can optionally set the number of decimals to round to and print out.
    If a padding value is given, prefixes with enough spaces to make the final
    string at least as long as padding.
    """
    if decimals is not None:
        number = round(float(number), decimals)
        if decimals is 0:
            number = int(number)

    string = str(number)

    if decimals:
        if '.' not in string:
            string += '.'
        decimals_length = len(string[string.index('.') + 1:])
        zero_length = decimals - decimals_length
        string += '0' * zero_length

    if padding is not None and len(string) < padding:
        pad_length = padding - len(string)
        string = (' ' * pad_length) + string

    return string


def generate_all_equations(
    shuffle=True,
    max_count=None,
    padding=True,
):
    """
    A generator that generates all possible math equations given the global
    configuration.
    Okay, when there's multiple possible types of operations, this will not
    generate ALL the equations, but a good chunk of them.
    """
    # Generate all possible unique sets of numbers
    number_permutations = itertools.permutations(
        range(MIN_NUMBER, MAX_NUMBER + 1), N_OPERATIONS + 1
    )

    if shuffle:
        number_permutations = list(number_permutations)
        random.shuffle(number_permutations)

    if max_count is not None:
        number_permutations = itertools.islice(number_permutations, max_count)

    for numbers in number_permutations:
        numbers = [
            to_padded_string(
                n,
                padding=MAX_NUMBER_LENGTH_LEFT_SIDE,
                decimals=DECIMALS,
            )
            for n in numbers
        ]

        equation = numbers[0]
        for j in range(N_OPERATIONS):
            operation = random.choice(OPERATIONS)
            equation += ' {} {}'.format(operation, numbers[j + 1])

        yield to_padded_string(
            equation,
            padding=MAX_EQUATION_LENGTH,
        )


def build_dataset():
    """
    Builds a dataset based on the global config.
    Returns (x_test, y_test, x_train, y_train).
    """
    generator = generate_all_equations(max_count=N_EXAMPLES)

    n_test = round(SPLIT * N_EXAMPLES)
    n_train = N_EXAMPLES - n_test

    x_test = np.zeros(
        (n_test, MAX_EQUATION_LENGTH, N_FEATURES), dtype=np.bool
    )
    y_test = np.zeros(
        (n_test, MAX_RESULT_LENGTH, N_FEATURES), dtype=np.bool
    )

    for i, equation in enumerate(itertools.islice(generator, n_test)):
        result = to_padded_string(
            eval(equation),
            padding=MAX_RESULT_LENGTH,
            decimals=DECIMALS,
        )

        for t, char in enumerate(equation):
            x_test[i, t, encoder.char_to_one_hot_index(char)] = 1

        for t, char in enumerate(result):
            y_test[i, t, encoder.char_to_one_hot_index(char)] = 1

    x_train = np.zeros(
        (n_train, MAX_EQUATION_LENGTH, N_FEATURES), dtype=np.bool
    )
    y_train = np.zeros(
        (n_train, MAX_RESULT_LENGTH, N_FEATURES), dtype=np.bool
    )

    for i, equation in enumerate(generator):
        result = to_padded_string(
            eval(equation),
            padding=MAX_RESULT_LENGTH,
            decimals=DECIMALS,
        )

        for t, char in enumerate(equation):
            x_train[i, t, encoder.char_to_one_hot_index(char)] = 1

        for t, char in enumerate(result):
            y_train[i, t, encoder.char_to_one_hot_index(char)] = 1

    return x_test, y_test, x_train, y_train


def build_model():
    """
    Builds and returns the model based on the global config.
    """
    input_shape = (MAX_EQUATION_LENGTH, N_FEATURES)

    model = Sequential()

    # Encoder:
    model.add(LSTM(
        HIDDEN_SIZE,
        input_shape=input_shape,
        return_sequences=(ENCODER_DEPTH > 1),
    ))
    model.add(Dropout(DROPOUT))

    for i in range(1, ENCODER_DEPTH):
        model.add(LSTM(
            HIDDEN_SIZE,
            return_sequences=(i != ENCODER_DEPTH - 1)
        ))
        model.add(Dropout(DROPOUT))

    # Repeats the input n times
    model.add(RepeatVector(MAX_RESULT_LENGTH))

    # Decoder:
    for _ in range(DECODER_DEPTH):
        model.add(LSTM(
            HIDDEN_SIZE,
            return_sequences=True,
        ))
        model.add(Dropout(DROPOUT))

    model.add(TimeDistributed(Dense(N_FEATURES)))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )

    return model


def build_seq2seq_model():
    """
    Builds and returns the model based on the global config, but using the
    seq2seq library: https://github.com/farizrahman4u/seq2seq
    """
    import seq2seq
    from seq2seq.models import Seq2Seq

    model = Seq2Seq(
        input_dim=N_FEATURES,
        input_length=MAX_EQUATION_LENGTH,
        hidden_dim=HIDDEN_SIZE,
        output_length=MAX_RESULT_LENGTH,
        output_dim=N_FEATURES,
        depth=(ENCODER_DEPTH, DECODER_DEPTH),
    )

    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['accuracy'],
    )

    return model


def print_example_predictions(count, model, x_test, y_test):
    """
    Print some example predictions along with their target from the test set.
    """
    print('Examples:')

    prediction_indices = np.random.choice(
        x_test.shape[0], size=count, replace=False
    )
    predictions = model.predict(x_test[prediction_indices, :])

    for i in range(count):
        print('{} = {}   (expected: {})'.format(
            encoder.one_hot_to_string(x_test[prediction_indices[i]]),
            prediction_to_string(predictions[i]),
            encoder.one_hot_to_string(y_test[prediction_indices[i]]),
        ))

        # For the last one, let's visualize the activations
        if i == count - 1:
            print('Activations:')
            print_activations(model, x_test[prediction_indices[i]])
            print()


def main():
    model = build_model()

    model.summary()
    print()

    x_test, y_test, x_train, y_train = build_dataset()

    epoch_batch_size = 10

    print()
    print_example_predictions(5, model, x_test, y_test)
    print()

    try:
        for iteration in range(int(EPOCHS / epoch_batch_size)):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            model.fit(
                x_train, y_train,
                epochs=epoch_batch_size,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
            )
            sleep(0.01)

            print()
            print_example_predictions(5, model, x_test, y_test)
            print()

    except KeyboardInterrupt:
        print(' Got Sigint')
    finally:
        sleep(0.01)
        model.save('model.h5')

        print_example_predictions(20, model, x_test, y_test)


if __name__ == '__main__':
    main()
