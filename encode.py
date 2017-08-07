import numpy as np


class OneHotEncoder:
    def __init__(self, operations):
        self.operations = operations

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
        (including spaces) will be mapped to a single value.
        """
        if char.isdigit():
            return int(char)
        elif char in self.operations:
            return 10 + self.operations.index(char)
        elif char == '.':
            return 10 + len(self.operations)
        else:
            return 10 + len(self.operations) + 1

    def char_to_one_hot(self, char):
        """
        Given a char, encodes it as a one-hot vector.
        """
        length = 10 + len(self.operations) + 2
        return self.one_hot(self.char_to_one_hot_index(char), length)

    def one_hot_index_to_char(self, index):
        """
        Given an index, returns the character encoded with that index.
        Will only work with encoded digits or operations, everything else will
        return the space character.
        """
        if index <= 9:
            return str(index)

        index -= 10

        if index < len(self.operations):
            return self.operations[index]

        if index == len(self.operations):
            return '.'

        return ' '

    def one_hot_to_char(self, vector):
        """
        Given a one-hot vector, returns the encoded char.
        """
        indices = np.nonzero(vector == 1.)

        assert len(indices) == 1
        assert len(indices[0]) == 1

        return self.one_hot_index_to_char(indices[0][0])

    def one_hot_to_string(self, matrix):
        """
        Given a matrix of single one-hot encoded char vectors, returns the
        encoded string.
        """
        return ''.join(self.one_hot_to_char(vector) for vector in matrix)
