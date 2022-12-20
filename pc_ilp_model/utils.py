import typing
from collections import Counter

import numpy as np


def positionAndCovering(widths: typing.List[int], heights: typing.List[int],
                        board_width: int, board_height: int):
    """
    Generates the set of valid positions, and the correspondence matrix,
    where items can be placed into the strip. Groups elements by shape and computes their demand.

    widths: list[int]: The widths of the circuits.
    heights: list[int]: The heights of the circuits.
    board_width: int: The width of the board.
    board_height: int: The current height of the board.
    :return:
    valid_positions: list[list[int]]: The set of valid positions grouped by item
    correspondence_matrix: np.ndarray: The correspondence matrix
    demands: list[int]: The number of occurrences of each unique item (by shape)
    upper_bounds: list[int]: The number of times each unique item can be placed inside the strip
    """
    counter = Counter(zip(widths, heights))
    widths, heights = zip(*list(counter.keys()))
    widths = list(widths)
    heights = list(heights)
    demands = list(counter.values())
    n_circuits = len(widths)

    j_h_cm: typing.List[typing.List[int]] = list()
    valid_positions: typing.List[typing.List[int]] = list()
    upper_bounds: typing.List[int] = list()

    current_pos = 0
    for circuit_idx in range(n_circuits):
        horizontal_valid_range = range(0, board_width - widths[circuit_idx] + 1, 1)
        vertical_valid_range = range(0, board_height - heights[circuit_idx] + 1, 1)
        covered_idxs = []
        first_valid_pos = current_pos

        for vertical_idx in vertical_valid_range:
            for horizontal_idx in horizontal_valid_range:
                sub_list = []
                for covered_row_idx in range(heights[circuit_idx]):
                    start = (vertical_idx + covered_row_idx) * board_width + horizontal_idx
                    end = start + widths[circuit_idx]
                    sub_list += list(range(start, end))
                covered_idxs.append(sub_list)
                current_pos += 1

        j_h_cm += covered_idxs
        valid_positions.append(list(range(first_valid_pos, current_pos)))
        upper_bounds.append(len(valid_positions[-1]))

    correspondence_matrix = np.zeros((len(j_h_cm), board_width * board_height), dtype=int)
    for idx, valid_pos in enumerate(j_h_cm):
        correspondence_matrix[idx, valid_pos] = 1

    return n_circuits, widths, heights, valid_positions, correspondence_matrix, demands, upper_bounds


