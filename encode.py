import numpy as np


class OneHotEncoder:
    """
    Convert characters and strings to one hot vectors and back.
    Uses argmax, so you can also pass softmax one-hot vectors and it will
    return the value with the highest probability.
    """

    def __init__(self, operations, decimals=False):
        self.operations = operations
        self._chars = [str(n) for n in range(10)] + operations

        if decimals:
            self._chars.append('.')

        # Catch-all character
        self._chars.append(' ')
        self._default_i = len(self._chars) - 1

        self._one_hot_length = len(self._chars)

        self.c_to_i = {i: c for c, i in enumerate(self._chars)}
        self.i_to_c = {c: i for c, i in enumerate(self._chars)}

    def one_hot(self, index, length):
        """
        Generates a one-hot vector of the given length that's 1.0 at the given
        index.
        """
        assert index < length

        array = np.zeros(length)
        array[index] = 1.

        return array

    def char_to_one_hot_index(self, char):
        """
        Given a char, encodes it as an integer to be used in a one-hot vector.
        Will only work with digits and the operations we use, everything else
        (including spaces) is mapped to a single value.
        """
        return self.c_to_i.get(char, self._default_i)

    def char_to_one_hot(self, char):
        """
        Given a char, encodes it as a one-hot vector.
        """
        return self.one_hot(
            self.char_to_one_hot_index(char),
            self._one_hot_length
        )

    def one_hot_index_to_char(self, index):
        """
        Given an index, returns the character encoded with that index.
        Will only work with encoded digits or operations, everything else will
        return the space character.
        """
        return self.i_to_c.get(index, ' ')

    def one_hot_to_char(self, vector):
        """
        Given a one-hot vector or softmax distribution,
        returns the encoded char.
        """
        return self.one_hot_index_to_char(np.argmax(vector))

    def one_hot_to_string(self, matrix):
        """
        Given a matrix of single one-hot encoded char vectors, returns the
        encoded string.
        """
        return ''.join(self.one_hot_to_char(vector) for vector in matrix)
