import random
import itertools
from decimal import Decimal

import numpy as np
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


MIN_NUMBER = 0
MAX_NUMBER = 99
OPERATIONS = ['+']
N_OPERATIONS = 1
MAX_N_EXAMPLES = (MAX_NUMBER - MIN_NUMBER) ** (N_OPERATIONS + 1)
N_EXAMPLES = min(MAX_N_EXAMPLES, 100000)
N_FEATURES = 10 + len(OPERATIONS) + 1
MAX_EQUATION_LENGTH = len(str(MAX_NUMBER)) * 2 + 3
SPLIT = .3
EPOCHS = 400
BATCH_SIZE = 1


def generate_number(
    use_decimals,
):
    if use_decimals:
        randfloat = random.random()
        range = MAX_NUMBER - MIN_NUMBER
        return Decimal(MIN_NUMBER + randfloat * range)
    else:
        return Decimal(random.randint(MIN_NUMBER, MAX_NUMBER))


def generate_basic_math(
    number_base=10,
    use_decimals=False,
):
    # TODO Use number_base
    output = ''

    number = generate_number(MIN_NUMBER, MAX_NUMBER, use_decimals)
    output += str(number)

    for n in range(N_OPERATIONS):
        next_number = generate_number(MIN_NUMBER, MAX_NUMBER, use_decimals)
        operation = random.choice(OPERATIONS)

        output += ' {} {}'.format(operation, next_number)

    return output


def generate_all_basic_math(
    shuffle=True,
    max_count=100,
    pad=True,
):
    permutations = itertools.permutations(range(MIN_NUMBER, MAX_NUMBER + 1), 2)

    if shuffle:
        permutations = list(permutations)
        random.shuffle(permutations)

    pad_size = None
    if pad:
        pad_size = MAX_EQUATION_LENGTH

    i = 0

    for n1, n2 in permutations:
        if max_count and i >= max_count:
            return

        output = str(n1)
        operation = random.choice(OPERATIONS)
        output += ' {} {}'.format(operation, n2)

        if pad:
            while len(output) < pad_size:
                output += ' '

        yield output
        i += 1


def one_hot(index, n_classes):
    assert index < n_classes

    array = [0.] * index
    array += [1.]
    array += [0.] * (n_classes - index - 1)

    return np.array(array)


def char_to_one_hot(c):
    n_classes = 10 + len(OPERATIONS) + 1

    index = -1

    if c.isdigit():
        index = int(c)
    elif c in OPERATIONS:
        index = 10 + OPERATIONS.index(c)
    else:
        index = 10 + len(OPERATIONS)

    return one_hot(index, n_classes)


def str_to_one_hot(s):
    n_classes = 10 + len(OPERATIONS) + 1

    array = []

    for c in s:
        array += [char_to_one_hot(c)]

    return np.array(array)


def one_hot_to_char(v):
    indices = np.nonzero(v == 1.)

    if not len(indices) == 1:
        raise ValueError('Not a one-hot encoded vector')

    index = indices[0][0]

    if index <= 9:
        return str(index)

    index -= 10

    if index < len(OPERATIONS):
        return OPERATIONS[index]

    return ' '


def one_hot_to_str(m):
    return ''.join(one_hot_to_char(v) for v in m)


def build_dataset():
    generator = generate_all_basic_math(max_count=N_EXAMPLES)

    equations = [x for x in generator]

    n_test = round(SPLIT * N_EXAMPLES)
    n_train = N_EXAMPLES - n_test

    testX = np.array([str_to_one_hot(s) for s in equations[:n_test]])
    testY = np.array([eval(s) for s in equations[:n_test]])

    trainX = np.array([
        str_to_one_hot(s)
        for s in equations[n_test:(n_test + n_train)]
    ])
    trainY = np.array([eval(s) for s in equations[n_test:(n_test + n_train)]])

    return testX, testY, trainX, trainY


def build_model():
    input_shape = (MAX_EQUATION_LENGTH, N_FEATURES)

    model = Sequential()
    model.add(
        LSTM(
            2,
            input_shape=input_shape,
        )
    )

    model.add(
        Dense(
            1,
            activation='relu',
        )
    )

    # optimizer = Adam(
    #     # clipnorm=5.,
    # )

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def main():
    model = build_model()

    testX, testY, trainX, trainY = build_dataset()

    try:
        model.fit(
            trainX, trainY,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
            validation_data=(testX, testY),
            callbacks=[
                EarlyStopping(
                    patience=20,
                ),
            ]
        )
    except KeyboardInterrupt:
        print(' Got Sigint')
    finally:
        model.save('model.h5')

        print('\nSome examples:')
        predictions = model.predict(testX[:10])

        for i in range(len(predictions)):
            print('"{}" = {} ({})'.format(
                one_hot_to_str(testX[i]),
                round(predictions[i][0], 2),
                testY[i],
            ))


if __name__ == '__main__':
    main()
