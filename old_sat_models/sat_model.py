import typing
import time
import json

import numpy as np
from z3 import *
import tqdm

from order_encoding import get_order_encoding_variables, retrieve_v_value, \
    axiom_clauses, linear_expression_order_encoding


def non_overlapping_by_standard_linear_encoding(n_rectangles: int,
                                                x_order_encoding: typing.List[typing.List[z3.BoolRef]],
                                                y_order_encoding: typing.List[typing.List[z3.BoolRef]],
                                                x_lower_bounds: typing.List[int],
                                                x_upper_bounds: typing.List[int],
                                                y_lower_bounds: typing.List[int],
                                                y_upper_bounds: typing.List[int],
                                                widths: typing.List[int],
                                                heights: typing.List[int],
                                                lr: typing.List[typing.List[z3.BoolRef]],
                                                ud: typing.List[typing.List[z3.BoolRef]]):
    constraints = []
    print("Generating non overlapping constraints before starting the solver...")
    for i in tqdm.tqdm(range(n_rectangles)):
        for j in range(i + 1, n_rectangles):
            lr_i_j_rel = linear_expression_order_encoding([1, -1], [i, j], -1 * widths[i], x_lower_bounds,
                                                          x_upper_bounds, x_order_encoding)
            lr_j_i_rel = linear_expression_order_encoding([-1, 1], [i, j], -1 * widths[j], x_lower_bounds,
                                                          x_upper_bounds, x_order_encoding)
            for rel in lr_i_j_rel:
                constraints.append(simplify(Or(Not(lr[i][j]), rel)))
            for rel in lr_j_i_rel:
                constraints.append(simplify(Or(Not(lr[j][i]), rel)))

            ud_i_j_rel = linear_expression_order_encoding([1, -1], [i, j], -1 * heights[i], y_lower_bounds,
                                                          y_upper_bounds, y_order_encoding)
            ud_j_i_rel = linear_expression_order_encoding([-1, 1], [i, j], -1 * heights[j], y_lower_bounds,
                                                          y_upper_bounds, y_order_encoding)

            for rel in ud_i_j_rel:
                constraints.append(simplify(Or(Not(ud[i][j]), rel)))
            for rel in ud_j_i_rel:
                constraints.append(simplify(Or(Not(ud[j][i]), rel)))

            constraints.append(Or(lr[i][j], lr[j][i], ud[i][j], ud[j][i]))

    return constraints


def non_overlapping_constraints_paper(n_rectangles: int,
                                      board_width: int,
                                      board_height: int,
                                      x_order_encoding: typing.List[typing.List[z3.BoolRef]],
                                      y_order_encoding: typing.List[typing.List[z3.BoolRef]],
                                      widths: typing.List[int],
                                      heights: typing.List[int],
                                      lr: typing.List[typing.List[z3.BoolRef]],
                                      ud: typing.List[typing.List[z3.BoolRef]]):
    constraints = []

    print("Generating non overlapping constraints before starting the solver...")
    for i in tqdm.tqdm(range(n_rectangles)):
        for j in range(i + 1, n_rectangles):
            for e in range(-1, board_width - widths[i]):
                if e + widths[i] <= board_width - widths[j]:
                    constraints.append(Or(Not(lr[i][j]), x_order_encoding[i][e + 1],
                                          Not(x_order_encoding[j][e + 1 + widths[i]])))
            for e in range(-1, board_width - widths[j]):
                if e + widths[j] <= board_width - widths[i]:
                    constraints.append(Or(Not(lr[j][i]), x_order_encoding[j][e + 1],
                                          Not(x_order_encoding[i][e + 1 + widths[j]])))

            for f in range(-1, board_height - heights[i]):
                if f + heights[i] <= board_height - heights[j]:
                    constraints.append(Or(Not(ud[i][j]), y_order_encoding[i][f + 1],
                                          Not(y_order_encoding[j][f + 1 + heights[i]])))
            for f in range(-1, board_height - heights[j]):
                if f + heights[j] <= board_height - heights[i]:
                    constraints.append(Or(Not(ud[j][i]), y_order_encoding[j][f + 1],
                                          Not(y_order_encoding[i][f + 1 + heights[j]])))

            constraints.append(Or(lr[i][j], lr[j][i], ud[i][j], ud[j][i]))

    return constraints


def ys_less_than_h_constraints(n_rectangles: int, height_lower_bound: int, height_upper_bound: int,
                               y_order_encoding: typing.List[typing.List[z3.BoolRef]],
                               h_order_encoding: typing.List[typing.List[z3.BoolRef]],
                               heights: typing.List[int]):
    constraints = []
    for i in range(n_rectangles):
        for o in range(height_lower_bound, height_upper_bound):
            constraints.append(Or(Not(h_order_encoding[0][o - height_lower_bound + 1]),
                                  y_order_encoding[i][o - heights[i] + 1]))

    return constraints


def large_rectangles_constraints_width(n_rectangles: int, board_width: int, widths: typing.List[int],
                                       lr: typing.List[typing.List[z3.BoolRef]]):
    constraints = []
    for i in range(n_rectangles):
        for j in range(i + 1, n_rectangles):
            if widths[i] + widths[j] > board_width:
                constraints.append(Not(lr[i][j]))
                constraints.append(Not(lr[j][i]))

    return constraints


def large_rectangles_constraints_height(n_rectangles: int, o: int, heights: typing.List[int],
                                        ud: typing.List[typing.List[z3.BoolRef]]):
    constraints = []
    for i in range(n_rectangles):
        for j in range(i + 1, n_rectangles):
            if heights[i] + heights[j] > o:
                constraints.append(Not(ud[i][j]))
                constraints.append(Not(ud[j][i]))

    return constraints


def same_rectangles_constraints(n_rectangles: int, widths: typing.List[int], heights: typing.List[int],
                                lr: typing.List[typing.List[z3.BoolRef]], ud: typing.List[typing.List[z3.BoolRef]]):
    constraints = []
    for i in range(n_rectangles):
        for j in range(i + 1, n_rectangles):
            if widths[i] == widths[j] and heights[i] == heights[j]:
                constraints.append(Not(lr[j][i]))
                constraints.append(Or(lr[i][j], Not(ud[j][i])))

    return constraints


def exclusive_constraints(n_rectangles: int, lr: typing.List[typing.List[z3.BoolRef]],
                          ud: typing.List[typing.List[z3.BoolRef]]):
    constraints = []
    for i in range(n_rectangles):
        for j in range(i + 1, n_rectangles):
            constraints.append(Or(Not(lr[i][j]), Not(lr[j][i])))
            constraints.append(Or(Not(ud[i][j]), Not(ud[j][i])))

    return constraints


def domain_reduction_constraints(n_rectangles: int,
                                 board_width: int,
                                 o: int,
                                 x_order_encoding: typing.List[typing.List[z3.BoolRef]],
                                 y_order_encoding: typing.List[typing.List[z3.BoolRef]],
                                 widths: typing.List[int],
                                 heights: typing.List[int],
                                 lr: typing.List[typing.List[z3.BoolRef]],
                                 ud: typing.List[typing.List[z3.BoolRef]]):
    constraints = []
    d = np.argmax(widths)
    new_x_d_upper_bound = math.floor((board_width - widths[d]) / 2)
    new_y_d_upper_bound = math.floor((o - heights[d]) / 2)

    for e in range(new_x_d_upper_bound, board_width - widths[d]):
        constraints.append(x_order_encoding[d][e + 1])
    for f in range(new_y_d_upper_bound, o - heights[d]):
        constraints.append(y_order_encoding[d][f + 1])

    for i in range(n_rectangles):
        if widths[i] > new_x_d_upper_bound:
            constraints.append(Not(lr[i][d]))
        if heights[i] > new_y_d_upper_bound:
            constraints.append(Not(ud[i][d]))

    return constraints


def get_initial_model(n_rectangles: int, board_width: int, height_lower_bound: int, height_upper_bound: int,
                      widths: typing.List[int], heights: typing.List[int]):
    x_lower_bounds = [0] * n_rectangles
    x_upper_bounds = [board_width - widths[i] for i in range(n_rectangles)]

    y_lower_bounds = [0] * n_rectangles
    y_upper_bounds = [height_upper_bound - heights[i] for i in range(n_rectangles)]

    x_order_encoding = get_order_encoding_variables("x", n_rectangles, x_lower_bounds, x_upper_bounds)
    y_order_encoding = get_order_encoding_variables("y", n_rectangles, y_lower_bounds, y_upper_bounds)

    lr = [[Bool(f"lr_{i + 1}_{j + 1}") for j in range(n_rectangles)] for i in range(n_rectangles)]
    ud = [[Bool(f"ud_{i + 1}_{j + 1}") for j in range(n_rectangles)] for i in range(n_rectangles)]

    x_axiom_clauses = axiom_clauses(list(range(n_rectangles)), x_lower_bounds, x_upper_bounds, x_order_encoding)
    print("x_axiom_clauses", x_axiom_clauses)
    y_axiom_clauses = axiom_clauses(list(range(n_rectangles)), y_lower_bounds, y_upper_bounds, y_order_encoding)
    print("y_axiom_clauses", y_axiom_clauses)

    non_overlapping_constraints_ = non_overlapping_by_standard_linear_encoding(n_rectangles,
                                                                               x_order_encoding,
                                                                               y_order_encoding,
                                                                               x_lower_bounds,
                                                                               x_upper_bounds,
                                                                               y_lower_bounds,
                                                                               y_upper_bounds,
                                                                               widths,
                                                                               heights,
                                                                               lr,
                                                                               ud)
    print("non_overlapping_constraints_", non_overlapping_constraints_)

    h_order_encoding = get_order_encoding_variables("h", 1, [height_lower_bound], [height_upper_bound])
    h_axiom_clauses = axiom_clauses([0], [height_lower_bound], [height_upper_bound], h_order_encoding)
    y_less_than_h_constraints_ = ys_less_than_h_constraints(n_rectangles, height_lower_bound, height_upper_bound,
                                                            y_order_encoding, h_order_encoding, heights)

    # Search space pruning techniques
    same_rectangles_constraints_ = same_rectangles_constraints(n_rectangles, widths, heights, lr, ud)
    large_rectangles_constraints_width_ = large_rectangles_constraints_width(n_rectangles, board_width, widths, lr)
    exclusive_constraints_ = exclusive_constraints(n_rectangles, lr, ud)

    solver = Solver()
    solver.add(x_axiom_clauses)
    solver.add(y_axiom_clauses)
    solver.add(non_overlapping_constraints_)
    solver.add(h_axiom_clauses)
    solver.add(y_less_than_h_constraints_)
    solver.add(same_rectangles_constraints_)
    solver.add(large_rectangles_constraints_width_)
    solver.add(exclusive_constraints_)

    return x_lower_bounds, x_upper_bounds, y_lower_bounds, y_upper_bounds, x_order_encoding, y_order_encoding, \
        h_order_encoding, lr, ud, solver


def solve_2ssp_instance(n_rectangles: int, board_width: int, height_lower_bound: int, height_upper_bound: int,
                        widths: typing.List[int], heights: typing.List[int], timeout: int = 300000):
    x_lower_bounds, x_upper_bounds, y_lower_bounds, y_upper_bounds, x_order_encoding, y_order_encoding, \
        h_order_encoding, lr, ud, solver = get_initial_model(n_rectangles, board_width, height_lower_bound,
                                                             height_upper_bound, widths, heights)

    # Bisection Method

    model = None
    lb = height_lower_bound
    ub = height_upper_bound if height_lower_bound < height_upper_bound else height_upper_bound + 1

    start = time.perf_counter()
    elapsed = 0

    while lb < ub:
        o = int((lb + ub) / 2)
        print("Testing satisfiability with height equal to %d..." % o)
        print(f"{timeout}ms remains")
        solver.push()

        large_rectangles_constraints_height_ = large_rectangles_constraints_height(n_rectangles, o, heights, ud)
        domain_reduction_constraints_ = domain_reduction_constraints(n_rectangles, board_width, o, x_order_encoding,
                                                                     y_order_encoding, widths, heights, lr, ud)
        solver.add(large_rectangles_constraints_height_)
        solver.add(domain_reduction_constraints_)
        solver.add(h_order_encoding[0][o - height_lower_bound + 1])

        if timeout > 0:
            solver.set(timeout=timeout)
        else:
            print("Reached timeout. Terminating...")
            break

        result = solver.check()
        elapsed = int((time.perf_counter() - start) * 1000)
        timeout -= elapsed

        if result == sat:
            print(f"It was satisfiable with height equal to {o}. {elapsed}ms passed.")
            model = solver.model()
        elif result == unsat:
            print(f"It was not satisfiable with height equal to {o}. {elapsed}ms passed.")
        elif result == unknown:
            print("Reached timeout. Terminating...")
            break

        solver.pop()
        if result == sat:
            ub = o
            solver.add(h_order_encoding[0][o - height_lower_bound + 1])
        else:
            lb = o + 1
            solver.add(Not(h_order_encoding[0][o - height_lower_bound + 1]))

    if model is None:
        print("Unable to find a solution to the problem")
        return -1, None, None, timeout, False

    xs = [
        retrieve_v_value(x_order_encoding, x_lower_bounds, x_upper_bounds, x_index, model)
        for x_index in range(n_rectangles)
    ]
    ys = [
        retrieve_v_value(y_order_encoding, y_lower_bounds, y_upper_bounds, y_index, model)
        for y_index in range(n_rectangles)
    ]

    return ub, xs, ys, elapsed, lb == ub


def solve_2ssp_instance_json(json_instance_filepath: str, timeout: int):
    instance = json.load(open(json_instance_filepath))

    ub, xs, ys, elapsed, exactly_solved = solve_2ssp_instance(instance["n_circuits"], instance["board_width"],
                                                              instance["height_lower_bound"],
                                                              instance["height_upper_bound"],
                                                              instance["widths"], instance["heights"], timeout)

    instance['board_height'] = ub
    instance['circuit_x'] = xs
    instance['circuit_y'] = ys

    return instance, elapsed, exactly_solved
