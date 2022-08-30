import numpy as np


def lower_bound(widths: np.ndarray, heights: np.ndarray, board_width: int):
    area_bound = np.dot(widths, heights) / board_width  # >= 1/4 Opt(I) (valid also for rotation)
    max_h = np.max(heights)

    return max(area_bound, max_h)


def nfdh_upper_bound(widths: np.ndarray, heights: np.ndarray, board_width: int):
    h_levels = [0]
    w_levels = [0]
    i = 0
    widths_by_non_increasing_heights = widths[heights.argsort()[::-1]]
    heights = np.sort(heights)[::-1]
    h_levels[-1] = heights[i]
    w_levels[-1] = widths_by_non_increasing_heights[i]
    for i in range(1, len(widths)):
        if board_width - w_levels[-1] >= widths[i]:
            w_levels[-1] += widths[i]
        else:
            w_levels.append(0)
            h_levels.append(0)
            w_levels[-1] = widths[i]
            h_levels[-1] = h_levels[-2] + heights[i]

    return h_levels[-1]


def ffdh_upper_bound(widths: np.ndarray, heights: np.ndarray, board_width: int):
    h_levels = [0]
    w_levels = [0]
    i = 0
    widths_by_non_increasing_heights = widths[heights.argsort()[::-1]]
    heights = np.sort(heights)[::-1]
    h_levels[-1] = heights[i]
    w_levels[-1] = widths_by_non_increasing_heights[i]
    for i in range(1, len(widths)):
        for level in range(len(h_levels)):
            if board_width - w_levels[level] >= widths[i]:
                w_levels[level] += widths[i]
                break
        else:
            w_levels.append(0)
            h_levels.append(0)
            w_levels[-1] = widths[i]
            h_levels[-1] = h_levels[-2] + heights[i]

    return h_levels[-1]


def bfdh_upper_bound(widths: np.ndarray, heights: np.ndarray, board_width: int):
    h_levels = [0]
    w_levels = [0]
    i = 0
    widths_by_non_increasing_heights = widths[heights.argsort()[::-1]]
    heights = np.sort(heights)[::-1]
    h_levels[-1] = heights[i]
    w_levels[-1] = widths_by_non_increasing_heights[i]
    for i in range(1, len(widths)):
        best_level = -1
        min_residual_horizontal_space = board_width
        for level in range(len(h_levels)):
            if board_width - w_levels[level] >= widths[i] and \
                    min_residual_horizontal_space > board_width - w_levels[level] - widths[i]:
                best_level = level
                min_residual_horizontal_space = board_width - w_levels[level] - widths[i]
        if best_level != -1:
            w_levels[best_level] += widths[i]
        else:
            w_levels.append(0)
            h_levels.append(0)
            w_levels[-1] = widths[i]
            h_levels[-1] = h_levels[-2] + heights[i]

    return h_levels[-1]


def kp01_upper_bound(widths: np.ndarray, heights: np.ndarray, board_width: int):
    h_level = 0
    unpacked_rectangles = list(heights.argsort()[::-1])
    while len(unpacked_rectangles) > 0:
        j_star = unpacked_rectangles[0]
        h_level += + heights[j_star]
        unpacked_rectangles.remove(j_star)
        # solve kp01 instance
        # maximize sum of h[i]*w[i]*chosen[i] for i in unpacked_rectangles
        # subject to sum of w[i] * chosen[i] <= W - w[j_star] with chosen[i] equal to 0 or 1
        areas = widths[unpacked_rectangles] * heights[unpacked_rectangles]
        max_area_sum, items_chosen = knapsack_01(board_width - widths[j_star], widths[unpacked_rectangles],
                                                 areas, len(unpacked_rectangles))

        to_be_removed = [unpacked_rectangles[i] for i in items_chosen]
        for item_to_remove in to_be_removed:
            unpacked_rectangles.remove(item_to_remove)

    return h_level


def knapsack_01(max_weight, weights, values, n_items):
    obj_fun_table = np.zeros(shape=(n_items + 1, max_weight + 1))

    for i in range(n_items + 1):
        for w in range(max_weight + 1):
            if i == 0 or w == 0:
                obj_fun_table[i, w] = 0
            elif weights[i - 1] <= w:
                obj_fun_table[i, w] = max(values[i - 1] + obj_fun_table[i - 1, w - weights[i - 1]],
                                          obj_fun_table[i - 1, w])
            else:
                obj_fun_table[i, w] = obj_fun_table[i - 1, w]

    max_value = obj_fun_table[n_items, max_weight]
    chosen_items = set()

    _max_weight = max_weight
    for n in range(n_items, 0, -1):
        if obj_fun_table[n, _max_weight] != obj_fun_table[n - 1, _max_weight]:
            chosen_items.add(n - 1)
            _max_weight = _max_weight - weights[n - 1]

    return max_value, chosen_items
