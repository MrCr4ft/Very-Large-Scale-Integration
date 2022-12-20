import typing

import numpy as np
import cplex
import docplex.mp
from docplex.mp.model import Model

from .utils import positionAndCovering


def buildModel(n_circuits: int, board_width: int, widths: typing.List[int],
               heights: typing.List[int], height_lower_bound: int,
               height_upper_bound: int, timeout: float):
    model = Model(name="position_and_covering_ilp_model")

    n_circuits, widths, heights, \
        valid_positions, correspondence_matrix, \
        demands, upper_bounds = positionAndCovering(widths, heights, board_width, height_lower_bound)

    new_indexes = []
    for i in range(n_circuits):
        for j in range(demands[i]):
            new_indexes.append(i)

    board_height = height_lower_bound

    x = {(i, j): model.binary_var(name='x_{0}_{1}'.format(i, j)) for i in range(n_circuits) for j in valid_positions[i]}

    for p in range(correspondence_matrix.shape[1]):
        model.add_constraint(model.sum(x[i, j] * correspondence_matrix[j, p] for i in range(n_circuits)
                                       for j in valid_positions[i]) <= 1)

    for i in range(n_circuits):
        model.add_constraint(model.sum(x[i, j] for j in valid_positions[i]) >= demands[i])

    for i in range(n_circuits):
        model.add_constraint(model.sum(x[i, j] for j in valid_positions[i]) <= upper_bounds[i])

    model.add_constraint(model.sum(x[i, j] * correspondence_matrix[j, p]
                                   for i in range(n_circuits)
                                   for j in valid_positions[i]
                                   for p in range(correspondence_matrix.shape[1])) <= board_width * board_height)

    solution = model.solve()
    assigned_positions = list(solution.as_df().name)
    assigned_positions = [int(s.split('_')[2]) for s in assigned_positions]
    xs = []
    ys = []

    for idx, position in enumerate(assigned_positions):
        index = np.argmax(correspondence_matrix[position, :])
        y_ = index // board_width
        x_ = index - (y_ * board_width)
        xs.append(x_)
        ys.append(board_height - y_ - heights[new_indexes[idx]])

    new_widths = []
    new_heights = []
    for idx in new_indexes:
        new_widths.append(widths[idx])
        new_heights.append(heights[idx])
    new_n_circuit = len(new_widths)

    return solution, new_n_circuit, new_widths, new_heights, board_height, xs, ys
