import typing

import pulp
import pulp as pl
from pulp import *
import numpy as np

from .utils import positionAndCovering


def buildModelPulp(board_width: int, widths: typing.List[int], heights: typing.List[int],
                   board_height: int):
    problem = pl.LpProblem(name="position_and_covering_milp_model")

    n_circuits, widths, heights, \
        valid_positions, correspondence_matrix, \
        demands, upper_bounds = positionAndCovering(widths, heights, board_width, board_height)

    new_indexes = []
    for i in range(n_circuits):
        for j in range(demands[i]):
            new_indexes.append(i)

    x = LpVariable.dicts("x", [(i, j) for i in range(n_circuits) for j in valid_positions[i]], 0, 1, LpBinary)

    print("Adding Constraints To The Model...")
    print("1")
    for p in range(correspondence_matrix.shape[1]):
        problem += lpSum(x[(i, j)] * correspondence_matrix[j, p] for i in range(n_circuits)
                         for j in valid_positions[i]) <= 1
    print("2")
    for i in range(n_circuits):
        problem += lpSum(x[(i, j)] for j in valid_positions[i]) >= demands[i]

    print("3")
    for i in range(n_circuits):
        problem += lpSum(x[(i, j)] for j in valid_positions[i]) <= upper_bounds[i]

    print("4")
    problem += lpSum(x[(i, j)] * correspondence_matrix[j, p]
                     for i in range(n_circuits)
                     for j in valid_positions[i]
                     for p in range(correspondence_matrix.shape[1])) <= board_width * board_height

    print("Constraints Added!")

    solver = pl.CPLEX_PY(msg=True)

    print("Model solving...")
    problem.solve(solver)

    assigned_positions = []

    for i in range(n_circuits):
        for j in valid_positions[i]:
            if x[(i, j)].varValue == 1:
                assigned_positions.append(j)
                break

    print(assigned_positions)

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

    return new_n_circuit, new_widths, new_heights, board_height, xs, ys