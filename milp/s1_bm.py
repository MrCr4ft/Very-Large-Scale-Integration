import typing
import math

import gurobipy as gp
from gurobipy import GRB


def S1_BM_Model(n_circuits: int, board_width: int, widths: typing.List[int],
                heights: typing.List[int], height_lower_bound: int,
                height_upper_bound: int, timeout_s: float) -> typing.Tuple[gp.MVar, gp.MVar, gp.Model]:
    model = gp.Model("S1_BM")
    model.setParam("TimeLimit", timeout_s)

    board_height = model.addVar(lb=height_lower_bound, ub=height_upper_bound, vtype=GRB.INTEGER, name="board_height")

    x = model.addMVar(shape=(n_circuits,), lb=[0] * n_circuits,
                      ub=[board_width - widths[i] for i in range(n_circuits)],
                      vtype=GRB.INTEGER, name="x")
    y = model.addMVar(shape=(n_circuits,), lb=[0] * n_circuits,
                      ub=[height_upper_bound - heights[i] for i in range(n_circuits)],
                      vtype=GRB.INTEGER, name="y")

    z_1 = model.addMVar(shape=(n_circuits, n_circuits), vtype=GRB.BINARY, name="z_1")
    z_2 = model.addMVar(shape=(n_circuits, n_circuits), vtype=GRB.BINARY, name="z_2")

    model.setObjective(board_height, GRB.MINIMIZE)

    model.addConstrs(((y[i] + heights[i] <= board_height) for i in range(n_circuits)), name="global_constraint")

    model.addConstrs((
        (x[i] + widths[i] <= x[j] + board_width * (1 - z_1[i, j]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="i_left_to_j")

    model.addConstrs((
        (x[j] + widths[j] <= x[i] + board_width * (1 - z_1[j, i]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="j_left_to_i")

    model.addConstrs((
        (y[i] >= y[j] + heights[j] - height_upper_bound * (1 - z_2[i, j]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="i_above_j")

    model.addConstrs((
        (x[i] + widths[i] >= x[j] - board_width * (1 - z_2[i, j]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="i_not_left_j_when_i_above_j")

    model.addConstrs((
        (x[j] + widths[j] >= x[i] - board_width * (1 - z_2[i, j]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="i_not_right_j_when_i_above_j")

    model.addConstrs((
        (y[j] >= y[i] + heights[i] - height_upper_bound * (1 - z_2[j, i]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="j_above_i")

    model.addConstrs((
        (x[i] + widths[i] >= x[j] - board_width * (1 - z_2[j, i]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="i_not_left_j_when_j_above_i")

    model.addConstrs((
        (x[j] + widths[j] >= x[i] - board_width * (1 - z_2[j, i]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="i_not_right_j_when_j_above_i")

    model.addConstrs((
        z_1[i, j] + z_1[j, i] + z_2[i, j] + z_2[j, i] == 1
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="logic_proposition_only_one")

    """
    SYMMETRY BREAKING CONSTRAINTS
        
    areas = [widths[i] * heights[i] for i in range(n_circuits)]
    sorted_indexes = [i for _, i in sorted(zip(areas, range(n_circuits)), reverse=True)]
    
    biggest_rectangle_x_bound = (board_width - widths[sorted_indexes[0]]) // 2
    biggest_rectangle_y_bound = (height_upper_bound - heights[sorted_indexes[0]]) // 2

    model.addConstr(x[sorted_indexes[0]] <= biggest_rectangle_x_bound, name="dom_red_x")
    model.addConstrs(((z_1[i, sorted_indexes[0]] == 0) for i in range(n_circuits) if widths[i] >
                     biggest_rectangle_x_bound), name="dom_red_x_impl")

    model.addConstr(y[sorted_indexes[0]] <= biggest_rectangle_y_bound, name="dom_red_y")
    model.addConstrs(((z_2[sorted_indexes[0], i] == 0) for i in range(n_circuits) if heights[i] >
                     biggest_rectangle_y_bound), name="dom_red_y_impl")
    """

    model.params.MIPFocus = 1

    return x, y, model
