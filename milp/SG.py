import typing
import math

import gurobipy as gp
from gurobipy import GRB


def SG_BM_Model(n_circuits: int, board_width: int, widths: typing.List[int],
                 heights: typing.List[int], height_lower_bound: int,
                 height_upper_bound: int, timeout_s: float) -> typing.Tuple[gp.MVar, gp.MVar, gp.Model]:
    model = gp.Model("SG_BM")
    model.setParam("TimeLimit", timeout_s)

    board_height = model.addVar(lb=height_lower_bound, ub=height_upper_bound, vtype=GRB.INTEGER, name="board_height")

    x = model.addMVar(shape=(n_circuits,), lb=[0] * n_circuits,
                      ub=[height_upper_bound - widths[i] for i in range(n_circuits)],
                      vtype=GRB.INTEGER, name="x")
    y = model.addMVar(shape=(n_circuits,), lb=heights,
                      ub=[board_width for i in range(n_circuits)],
                      vtype=GRB.INTEGER, name="y")

    z_1 = model.addMVar(shape=(n_circuits, n_circuits), vtype=GRB.BINARY, name="z_1")
    z_2 = model.addMVar(shape=(n_circuits, n_circuits), vtype=GRB.BINARY, name="z_2")

    model.setObjective(board_height, GRB.MINIMIZE)

    model.addConstrs((x[i] + widths[i] <= board_height for i in range(n_circuits)),
                     name="effective_height_bound")

    model.addConstrs((
        (x[i] + widths[i] <= x[j] + height_upper_bound * (1 - z_1[i, j]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="c1")

    model.addConstrs((
        (x[j] + widths[j] <= x[i] + height_upper_bound * (1 - z_1[j, i]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="c2")

    model.addConstrs((
        (y[i] - heights[i] >= y[j] - board_width * (1 - z_2[i, j]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="c3")

    model.addConstrs((
        (y[j] - heights[j] >= y[i] - board_width * (1 - z_2[j, i]))
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="c4")

    model.addConstrs((
        z_1[i, j] + z_1[j, i] + z_2[i, j] + z_2[j, i] == 1
        for i in range(n_circuits)
        for j in range(i + 1, n_circuits)
    ), name="c5")

    model.params.MIPFocus = 1

    return x, y, model
