import typing
from time import perf_counter
from collections import Counter
from abc import ABC, abstractmethod

import pulp as pl
from numpy import ndarray
from pulp import *
import numpy as np

SOLVERS = {
    "GUROBI_CMD": lambda time_limit: pl.GUROBI_CMD(msg=True, timeLimit=time_limit, options=[("MIPFocus", 1)]),
    "GUROBI_PY": lambda time_limit: pl.GUROBI(msg=True, timeLimit=time_limit),
    "CPLEX_CMD": lambda time_limit: pl.CPLEX_CMD(msg=True, timeLimit=time_limit,
                                             options=["set mip tolerances integrality 1e-05", "set emphasis mip 1"]),
    "GLPK": lambda time_limit: pl.GLPK_CMD(msg=True, timeLimit=time_limit),
    "PULP_CBC": lambda time_limit: pl.PULP_CBC_CMD(msg=True, timeLimit=time_limit),
    "CPLEX_PY": lambda time_limit: pl.CPLEX_PY(msg=True, timeLimit=time_limit),
}


def valid_positions_for_circuit(board_width: int, board_height: int, circuit_width: int, circuit_height: int) \
        -> list[list[int]]:
    covered_points_for_position = []
    horizontal_valid_range = range(0, board_width - circuit_width + 1, 1)
    vertical_valid_range = range(0, board_height - circuit_height + 1, 1)
    for i in horizontal_valid_range:
        for j in vertical_valid_range:
            covered_points = []
            for k in range(circuit_height):
                start = (j + k) * board_width + i
                end = start + circuit_width
                covered_points += list(range(start, end))
            covered_points_for_position.append(covered_points)

    return covered_points_for_position


class PCMILPProblem(ABC):
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int):
        self.n_circuits = n_circuits
        self.board_width = board_width
        self.widths = widths
        self.heights = heights
        self.height_lower_bound = height_lower_bound
        self.height_upper_bound = height_upper_bound
        self.time_limit = 300
        self.set_solver("GUROBI_CMD")
        print("Solver set to GUROBI CMD by default. Change solver if needed.")
        self.__compute_circuits_demands()

    def __compute_circuits_demands(self):
        counter = Counter(zip(self.widths, self.heights))
        widths, heights = zip(*list(counter.keys()))
        self.widths, self.heights, self.demands = list(widths), list(heights), list(counter.values())
        self.n_unique_circuits = len(widths)

    @staticmethod
    @abstractmethod
    def from_instance_json(json_filepath: str) -> "PCMILPProblem":
        pass

    @staticmethod
    @abstractmethod
    def from_dict(instance_dict: dict) -> "PCMILPProblem":
        pass

    def set_solver(self, solver_name: str) -> None:
        self.solver = SOLVERS[solver_name](self.time_limit)
        print("Solver set to {}".format(solver_name))

    def set_time_limit(self, time_limit: int) -> None:
        self.time_limit = time_limit
        print("Time limit set to {}".format(time_limit))

    def _set_board_height(self, board_height: int) -> None:
        self.board_height = board_height

    @abstractmethod
    def _positioning_and_covering_step(self, *args, **kwargs) \
            -> typing.Tuple[typing.List[typing.List[int]], typing.Optional[typing.List[typing.List[int]]],
                            np.ndarray, typing.List[int]]:
        """
        Generates the set of valid positions, and the correspondence matrix,
        where items can be placed into the strip.

        widths: list[int]: The widths of the circuits.
        heights: list[int]: The heights of the circuits.
        board_width: int: The width of the board.
        board_height: int: The current height of the board.
        rotated: bool: Whether to rotate the circuits or not.
        :return:
        valid_positions: list[list[int]]: The set of valid positions grouped by item
        correspondence_matrix: np.ndarray: The correspondence matrix
        demands: list[int]: The number of occurrences of each unique item (by shape)
        upper_bounds: list[int]: The number of times each unique item can be placed inside the strip
        """
        pass

    @abstractmethod
    def _build_model(self, *args, **kwargs) -> typing.Tuple[pl.LpProblem, typing.Dict[typing.Tuple, pl.LpVariable],
                                                            typing.Optional[typing.Dict[typing.Tuple, pl.LpVariable]]]:
        pass

    @abstractmethod
    def _retrieve_solution(self, *args, **kwargs) \
            -> tuple[int, list[int], list[int], list[int], list[int]]:
        pass

    @abstractmethod
    def solve(self, *args, **kwargs) -> typing.Dict[str, typing.Any]:
        pass


class PCMILPProblemNoRotation(PCMILPProblem):
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int):
        super().__init__(n_circuits, board_width, widths, heights, height_lower_bound,
                         height_upper_bound)

    @staticmethod
    def from_instance_json(json_filepath: str) -> "PCMILPProblemNoRotation":
        with open(json_filepath) as instance_file:
            instance = json.load(instance_file)
            return PCMILPProblemNoRotation(**instance)

    @staticmethod
    def from_dict(instance_dict: dict) -> "PCMILPProblemNoRotation":
        return PCMILPProblemNoRotation(**instance_dict)

    def _positioning_and_covering_step(self) \
            -> typing.Tuple[typing.List[typing.List[int]], np.ndarray, typing.List[int]]:
        """
        Generates the set of valid positions, and the correspondence matrix,
        where items can be placed into the strip.
        widths: list[int]: The widths of the circuits.
        heights: list[int]: The heights of the circuits.
        board_width: int: The width of the board.
        board_height: int: The current height of the board.
        rotated: bool: Whether to rotate the circuits or not.
        :return:
        valid_positions: list[list[int]]: The set of valid positions grouped by item
        correspondence_matrix: np.ndarray: The correspondence matrix
        demands: list[int]: The number of occurrences of each unique item (by shape)
        upper_bounds: list[int]: The number of times each unique item can be placed inside the strip
        """
        j_h_cm: typing.List[typing.List[int]] = list()
        unique_labels: typing.List[typing.List[int]] = list()
        upper_bounds: typing.List[int] = list()

        unique_labels_counter = 0
        for circuit_idx in range(self.n_unique_circuits):
            valid_positions_ = valid_positions_for_circuit(self.board_width, self.board_height,
                                                           self.widths[circuit_idx], self.heights[circuit_idx])
            unique_labels.append(list(range(unique_labels_counter, unique_labels_counter + len(valid_positions_))))
            unique_labels_counter += len(valid_positions_)
            upper_bounds.append(len(valid_positions_))
            j_h_cm += valid_positions_

        correspondence_matrix = np.zeros((len(j_h_cm), self.board_width * self.board_height), dtype=int)
        for idx, valid_pos in enumerate(j_h_cm):
            correspondence_matrix[idx, valid_pos] = 1

        return unique_labels, correspondence_matrix, upper_bounds

    def _build_model(self, valid_positions: typing.List[typing.List[int]], correspondence_matrix: np.ndarray,
                     upper_bounds: typing.List[int]) \
            -> typing.Tuple[pl.LpProblem, typing.Dict[typing.Tuple, pl.LpVariable]]:
        model = pl.LpProblem("position_and_covering_milp_model")
        x = LpVariable.dicts("x", [(i, j) for i in range(self.n_unique_circuits) for j in valid_positions[i]], 0, 1,
                             LpInteger)

        for p in range(correspondence_matrix.shape[1]):
            variables_to_sum = [x[i, j] for i in range(self.n_unique_circuits) for j in valid_positions[i] if
                                correspondence_matrix[j, p] == 1]
            model += lpSum(variables_to_sum) <= 1

        for i in range(self.n_unique_circuits):
            model += lpSum(x[(i, j)] for j in valid_positions[i]) >= self.demands[i]

        for i in range(self.n_unique_circuits):
            model += lpSum(x[(i, j)] for j in valid_positions[i]) <= upper_bounds[i]

        variables_to_sum = [x[i, j] for i in range(self.n_unique_circuits) for j in valid_positions[i]
                            for p in range(correspondence_matrix.shape[1]) if correspondence_matrix[j, p] == 1]
        model += lpSum(variables_to_sum) <= self.board_width * self.board_height

        return model, x

    def _retrieve_solution(self, x: typing.Dict[typing.Tuple, LpVariable],
                           valid_positions: [typing.List[typing.List[int]]], correspondence_matrix: np.ndarray) \
            -> tuple[int, list[int], list[int], list[int], list[int]]:
        assigned_positions = []
        for i in range(self.n_unique_circuits):
            for j in valid_positions[i]:
                if x[(i, j)].roundedValue() == 1:
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

    def solve(self) -> typing.Dict[str, typing.Any]:
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
            start_time = perf_counter()
            model.solve(self.solver)
            end_time = perf_counter()

            time_limit_exceeded = np.ceil(end_time - start_time) >= self.time_limit

            print("Accessing to the status of the model...")
            print("The status of the model is %s" % pl.LpStatus[model.status])

            if model.status == 1:
                print("Model solved")
                lb = ub + 1
            elif time_limit_exceeded:
                print("Time limit exceeded")
                ub = lb - 1
            else:
                print("Unsatisfiable with height equal to %d" % lb)
                lb += 1

        if time_limit_exceeded:
            return None

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


class PCMILPProblemRotation(PCMILPProblem):
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int):
        super().__init__(n_circuits, board_width, widths, heights, height_lower_bound,
                         height_upper_bound)

    @staticmethod
    def from_instance_json(json_filepath: str) -> "PCMILPProblemRotation":
        with open(json_filepath) as instance_file:
            instance = json.load(instance_file)
            return PCMILPProblemRotation(**instance)

    @staticmethod
    def from_dict(instance_dict: dict) -> "PCMILPProblemRotation":
        return PCMILPProblemRotation(**instance_dict)

    def _positioning_and_covering_step(self) \
            -> tuple[list[list[int]], list[list[int]], ndarray, list[int]]:
        """
        Generates the set of valid positions, and the correspondence matrix,
        where items can be placed into the strip when rotation is allowed.
        widths: list[int]: The widths of the circuits.
        heights: list[int]: The heights of the circuits.
        board_width: int: The width of the board.
        board_height: int: The current height of the board.
        rotated: bool: Whether to rotate the circuits or not.
        :return:
        valid_positions: list[list[int]]: The set of valid positions grouped by item
        valid_positions_rotated: list[list[int]]: The set of valid positions grouped by item when rotated
        correspondence_matrix: np.ndarray: The correspondence matrix
        demands: list[int]: The number of occurrences of each unique item (by shape)
        upper_bounds: list[int]: The number of times each unique item can be placed inside the strip
        """
        j_h_cm: typing.List[typing.List[int]] = list()
        unique_labels: typing.List[typing.List[int]] = list()
        upper_bounds: typing.List[int] = list()

        unique_labels_counter = 0
        for circuit_idx in range(self.n_unique_circuits):
            valid_positions_ = valid_positions_for_circuit(self.board_width, self.board_height,
                                                           self.widths[circuit_idx], self.heights[circuit_idx])
            unique_labels.append(list(range(unique_labels_counter, unique_labels_counter + len(valid_positions_))))
            unique_labels_counter += len(valid_positions_)
            upper_bounds.append(len(valid_positions_))
            j_h_cm += valid_positions_

        last_label_index_not_rotated = len(unique_labels)
        for circuit_idx in range(self.n_unique_circuits):
            if self.widths[circuit_idx] == self.heights[circuit_idx]:
                unique_labels.append([])
                continue
            valid_positions_ = valid_positions_for_circuit(self.board_width, self.board_height,
                                                           self.heights[circuit_idx], self.widths[circuit_idx])
            unique_labels.append(list(range(unique_labels_counter, unique_labels_counter + len(valid_positions_))))
            unique_labels_counter += len(valid_positions_)
            upper_bounds.append(len(valid_positions_))
            j_h_cm += valid_positions_

        correspondence_matrix = np.zeros((len(j_h_cm), self.board_width * self.board_height), dtype=int)
        for idx, valid_pos in enumerate(j_h_cm):
            correspondence_matrix[idx, valid_pos] = 1

        return unique_labels[:last_label_index_not_rotated], \
               unique_labels[last_label_index_not_rotated:], correspondence_matrix, upper_bounds

    def _build_model(self, valid_positions: typing.List[typing.List[int]], correspondence_matrix: np.ndarray,
                     upper_bounds: typing.List[int], valid_positions_rotated: typing.List[typing.List[int]]) \
            -> typing.Tuple[pl.LpProblem, typing.Dict[typing.Tuple, pl.LpVariable],
                            typing.Dict[typing.Tuple, pl.LpVariable]]:
        model = pl.LpProblem("position_and_covering_milp_model_rotation_allowed", pl.LpMinimize)
        model += 0, "objective_function"

        x = LpVariable.dicts("x", [(i, j) for i in range(self.n_unique_circuits) for j in valid_positions[i]],
                             0, 1, LpInteger)
        y = LpVariable.dicts("y", [(i, j) for i in range(self.n_unique_circuits) for j in valid_positions_rotated[i]],
                             0, 1, LpInteger)

        for p in range(correspondence_matrix.shape[1]):
            variables_to_sum_x = [x[i, j] for i in range(self.n_unique_circuits) for j in valid_positions[i] if
                                  correspondence_matrix[j, p] == 1]
            variables_to_sum_y = [y[i, j] for i in range(self.n_unique_circuits) for j in valid_positions_rotated[i] if
                                  correspondence_matrix[j, p] == 1]
            model += lpSum(variables_to_sum_x) + lpSum(variables_to_sum_y) <= 1

        for i in range(self.n_unique_circuits):
            model += lpSum(x[(i, j)] for j in valid_positions[i]) + \
                     lpSum(y[(i, j)] for j in valid_positions_rotated[i]) >= self.demands[i]

        for i in range(self.n_unique_circuits):
            model += lpSum(x[(i, j)] for j in valid_positions[i]) + \
                     lpSum(y[(i, j)] for j in valid_positions_rotated[i]) <= upper_bounds[i]

        variables_to_sum_x = [x[(i, j)] for i in range(self.n_unique_circuits) for j in valid_positions[i]
                              for p in range(correspondence_matrix.shape[1]) if correspondence_matrix[j, p] == 1]
        variables_to_sum_y = [y[(i, j)] for i in range(self.n_unique_circuits) for j in valid_positions_rotated[i]
                              for p in range(correspondence_matrix.shape[1]) if correspondence_matrix[j, p] == 1]
        model += lpSum(variables_to_sum_x) + lpSum(variables_to_sum_y) <= self.board_width * self.board_height

        return model, x, y

    def _retrieve_solution(self, x: typing.Dict[typing.Tuple, LpVariable], y: typing.Dict[typing.Tuple, LpVariable],
                           valid_positions: [typing.List[typing.List[int]]],
                           valid_positions_rotated: typing.List[typing.List[int]], correspondence_matrix: np.ndarray) \
            -> tuple[int, list[int], list[int], list[int], list[int]]:
        assigned_positions = []
        rotated = []
        for i in range(self.n_unique_circuits):
            for j in valid_positions[i]:
                if x[(i, j)].roundedValue() == 1:
                    assigned_positions.append(j)
                    rotated.append(False)
            for j in valid_positions_rotated[i]:
                if y[(i, j)].roundedValue() == 1:
                    assigned_positions.append(j)
                    rotated.append(True)

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
            if rotated[idx]:
                ys.append(self.board_height - y_ - self.widths[circuits_indexes[idx]])
            else:
                ys.append(self.board_height - y_ - self.heights[circuits_indexes[idx]])

        widths = []
        heights = []
        for idx in circuits_indexes:
            if rotated[idx]:
                widths.append(self.heights[idx])
                heights.append(self.widths[idx])
            else:
                widths.append(self.widths[idx])
                heights.append(self.heights[idx])

        return len(widths), widths, heights, xs, ys

    def solve(self) -> typing.Dict[str, typing.Any]:
        lb = self.height_lower_bound
        ub = self.height_upper_bound
        # Linear search for the optimal height
        while lb <= ub:
            self._set_board_height(lb)
            valid_positions, valid_positions_rotated, correspondence_matrix, upper_bounds = \
                self._positioning_and_covering_step()

            # Create the model
            start_time = perf_counter()

            print("Building model using Pulp...")
            model, x, y = self._build_model(valid_positions, correspondence_matrix, upper_bounds,
                                            valid_positions_rotated)

            end_time = perf_counter()
            print("It took %.2f seconds to build the model" % (end_time - start_time))

            # Solve the model
            print("Trying to solve the model with board height equal to %d..." % lb)
            print("The time limit is %d seconds" % self.time_limit)
            start_time = perf_counter()
            model.solve(self.solver)
            end_time = perf_counter()

            time_limit_exceeded = np.ceil(end_time - start_time) >= self.time_limit

            print("Accessing to the status of the model...")
            print("The status of the model is %s" % pl.LpStatus[model.status])

            if model.status == 1:
                print("Model solved")
                lb = ub + 1
            elif time_limit_exceeded:
                print("Time limit exceeded")
                ub = lb - 1
            else:
                print("Unsatisfiable with height equal to %d" % lb)
                lb += 1

        if time_limit_exceeded:
            return None

        # Get the solution
        n_circuits, widths, heights, xs, ys = self._retrieve_solution(x, y, valid_positions, valid_positions_rotated,
                                                                      correspondence_matrix)

        return {
            'board_width': self.board_width,
            'board_height': self.board_height,
            'n_circuits': n_circuits,
            'widths': widths,
            'heights': heights,
            'x': xs,
            'y': ys
        }
