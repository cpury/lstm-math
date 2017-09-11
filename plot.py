import numpy as np
import matplotlib.pyplot as plt


def plot_2d_space(model, one_hot_encoder, x, y, n=None, reverse=False):
    """
    For models trained on equations ala 'a + b', this will plot a scatter plot
    with correct examples in green and incorrect ones in red.
    """
    if n is None:
        n = len(x)

    order = -1 if reverse else 1

    predictions = model.predict(x[:n])

    correct_coords = []
    incorrect_coords = []

    for i, prediction in enumerate(predictions):
        target = y[i]

        equation_string = one_hot_to_string(x[i])[::order]
        prediction_string = one_hot_to_string(prediction)[::order]
        target_string = one_hot_to_string(target)[::order]

        equation_plus_index = equation_string.index('+')
        n1 = int(equation_string[:equation_plus_index-1])
        n2 = int(equation_string[equation_plus_index+1:])

        if prediction_string == target_string:
            # Correct
            correct_coords.append((n1, n2,))
        else:
            incorrect_coords.append((n1, n2,))

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

    correct_coords = np.array(correct_coords)
    ax.scatter(
        correct_coords[:, 0],
        correct_coords[:, 1],
        alpha=0.33, c='green', edgecolors='none', s=20, label='correct'
    )

    incorrect_coords = np.array(incorrect_coords)
    ax.scatter(
        incorrect_coords[:, 0],
        incorrect_coords[:, 1],
        alpha=0.33, c='red', edgecolors='none', s=20, label='incorrect'
    )

    plt.show()
