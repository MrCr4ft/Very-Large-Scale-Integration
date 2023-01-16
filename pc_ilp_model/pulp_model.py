import typing
from time import perf_counter
from collections import Counter

import pulp as pl
from pulp import *
import numpy as np

SOLVERS = {
    "GUROBI": lambda time_limit: pl.GUROBI_CMD(msg=True, timeLimit=time_limit),
    "CPLEX": lambda time_limit: pl.CPLEX_CMD(msg=True, timeLimit=time_limit),
    "GLPK": lambda time_limit: pl.GLPK_CMD(msg=True, timeLimit=time_limit),
    "PULP_CBC": lambda time_limit: pl.PULP_CBC_CMD(msg=True, timeLimit=time_limit)
}


class PositioningCoveringMILPProblem:
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int):
        self.n_circuits = n_circuits
        self.board_width = board_width
        self.widths = widths
        self.heights = heights
        self.height_lower_bound = height_lower_bound
        self.height_upper_bound = height_upper_bound
        self.time_limit = 300
        self.solver = "GUROBI"

        counter = Counter(zip(widths, heights))
        widths, heights = zip(*list(counter.keys()))
        self.widths, self.heights, self.demands = list(widths), list(heights), list(counter.values())
        self.n_unique_circuits = len(widths)

    @staticmethod
    def from_instance_json(json_filepath: str):
        with open(json_filepath) as instance_file:
            instance = json.load(instance_file)
            return PositioningCoveringMILPProblem(**instance)

    @staticmethod
    def from_dict(instance_dict: dict):
        return PositioningCoveringMILPProblem(**instance_dict)

    def set_solver(self, solver_name: str):
        self.solver = SOLVERS[solver_name](self.time_limit)
        print("Solver set to {}".format(solver_name))

    def set_time_limit(self, time_limit: int):
        self.time_limit = time_limit
        print("Time limit set to {}".format(time_limit))

    def _set_board_height(self, board_height: int):
        self.board_height = board_height

    def _positioning_and_covering_step(self):
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

        j_h_cm: typing.List[typing.List[int]] = list()
        valid_positions: typing.List[typing.List[int]] = list()
        upper_bounds: typing.List[int] = list()

        current_pos = 0
        for circuit_idx in range(self.n_unique_circuits):
            horizontal_valid_range = range(0, self.board_width - self.widths[circuit_idx] + 1, 1)
            vertical_valid_range = range(0, self.board_height - self.heights[circuit_idx] + 1, 1)
            covered_idxs = []
            first_valid_pos = current_pos

            for vertical_idx in vertical_valid_range:
                for horizontal_idx in horizontal_valid_range:
                    sub_list = []
                    for covered_row_idx in range(self.heights[circuit_idx]):
                        start = (vertical_idx + covered_row_idx) * self.board_width + horizontal_idx
                        end = start + self.widths[circuit_idx]
                        sub_list += list(range(start, end))
                    covered_idxs.append(sub_list)
                    current_pos += 1

            j_h_cm += covered_idxs
            valid_positions.append(list(range(first_valid_pos, current_pos)))
            upper_bounds.append(len(valid_positions[-1]))

        correspondence_matrix = np.zeros((len(j_h_cm), self.board_width * self.board_height), dtype=int)
        for idx, valid_pos in enumerate(j_h_cm):
            correspondence_matrix[idx, valid_pos] = 1

        return valid_positions, correspondence_matrix, upper_bounds

    def _build_model(self, valid_positions: typing.List[typing.List[int]],
                     correspondence_matrix: np.ndarray, upper_bounds: typing.List[int]):
        model = pl.LpProblem("position_and_covering_ilp_model")
        x = LpVariable.dicts("x", [(i, j) for i in range(self.n_unique_circuits) for j in valid_positions[i]], 0, 1,
                             LpBinary)

        for p in range(correspondence_matrix.shape[1]):
            model += lpSum(x[(i, j)] * correspondence_matrix[j, p] for i in range(self.n_unique_circuits)
                           for j in valid_positions[i]) <= 1

        for i in range(self.n_unique_circuits):
            model += lpSum(x[(i, j)] for j in valid_positions[i]) >= self.demands[i]

        for i in range(self.n_unique_circuits):
            model += lpSum(x[(i, j)] for j in valid_positions[i]) <= upper_bounds[i]

        model += lpSum(x[(i, j)] * correspondence_matrix[j, p]
                       for i in range(self.n_unique_circuits)
                       for j in valid_positions[i]
                       for p in range(correspondence_matrix.shape[1])) <= self.board_width * self.board_height

        return model, x

    def _retrieve_solution(self, x: typing.Dict[typing.Tuple, LpVariable],
                           valid_positions: [typing.List[typing.List[int]]], correspondence_matrix: np.ndarray):
        assigned_positions = []
        for i in range(self.n_unique_circuits):
            for j in valid_positions[i]:
                if x[(i, j)].value() == 1:
                    assigned_positions.append(j)

        xs = []
        ys = []

        circuits_indexes = []
        for i in range(self.n_unique_circuits):
            for j in range(self.demands[i]):
                circuits_indexes.append(i)

        for idx, position in enumerate(assigned_positions):
            index = np.argmax(correspondence_matrix[position, :])
            y_ = index // self.board_width
            x_ = index - (y_ * self.board_width)
            xs.append(x_)
            ys.append(self.board_height - y_ - self.heights[circuits_indexes[idx]])

        widths = []
        heights = []
        for idx in circuits_indexes:
            widths.append(self.widths[idx])
            heights.append(self.heights[idx])

        return len(widths), widths, heights, xs, ys

    def solve(self):
        lb = self.height_lower_bound
        ub = self.height_upper_bound
        # Linear search for the optimal height
        while lb <= ub:
            self._set_board_height(lb)
            valid_positions, correspondence_matrix, upper_bounds = self._positioning_and_covering_step()

            # Create the model
            start_time = perf_counter()

            print("Building model using Pulp...")
            model, x = self._build_model(valid_positions, correspondence_matrix, upper_bounds)

            end_time = perf_counter()
            print("It took %.2f seconds to build the model" % (end_time - start_time))

            # Solve the model
            print("Trying to solve the model with board height equal to %d..." % lb)
            print("The time limit is %d seconds" % self.time_limit)
            model.solve(self.solver)

            print("Accessing to the status of the model...")
            print("The status of the model is %s" % pl.LpStatus[model.status])

            if model.status == 1:
                print("Model solved")
                lb = ub + 1
            else:
                print("Unsatisfiable with height equal to %d" % lb)
                lb += 1

        # Get the solution
        n_circuits, widths, heights, xs, ys = self._retrieve_solution(x, valid_positions, correspondence_matrix)

        return {
            'board_width': self.board_width,
            'board_height': self.board_height,
            'n_circuits': n_circuits,
            'widths': widths,
            'heights': heights,
            'x': xs,
            'y': ys
        }


def dict_to_dzn(board_width: int, board_height: int, n_circuits: int, widths: typing.List[int],
                heights: typing.List[int], x: typing.List[int], y: typing.List[int]):
    dzn = "board_width = %d;\n" % board_width
    dzn += "board_height = %d;\n" % board_height
    dzn += "n_circuits = %d;\n" % n_circuits
    dzn += "widths = [%s];\n" % ", ".join([str(w) for w in widths])
    dzn += "heights = [%s];\n" % ", ".join([str(h) for h in heights])
    dzn += "x = [%s];\n" % ", ".join([str(x_) for x_ in x])
    dzn += "y = [%s];\n" % ", ".join([str(y_) for y_ in y])

    return dzn
