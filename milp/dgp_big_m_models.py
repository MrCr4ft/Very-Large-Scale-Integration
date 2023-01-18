import typing
from time import perf_counter
from abc import ABC, abstractmethod

import pulp as pl
from pulp import *
import numpy as np

from utils.heuristics import kp01_upper_bound

SOLVERS = {
    "GUROBI_CMD": lambda time_limit: pl.GUROBI_CMD(msg=True, timeLimit=time_limit, options=[("MIPFocus", 2)],
                                                   warmStart=True),
    "GUROBI_PY": lambda time_limit: pl.GUROBI(msg=True, timeLimit=time_limit),
    "CPLEX_CMD": lambda time_limit: pl.CPLEX_CMD(msg=True, timeLimit=time_limit,
                                                 options=["set mip tolerances integrality 1e-05",
                                                          "set emphasis mip 2"], warmStart=True),
    "GLPK": lambda time_limit: pl.GLPK_CMD(msg=True, timeLimit=time_limit),
    "PULP_CBC": lambda time_limit: pl.PULP_CBC_CMD(msg=True, timeLimit=time_limit),
    "CPLEX_PY": lambda time_limit: pl.CPLEX_PY(msg=True, timeLimit=time_limit),
}


class SPulpModel(ABC):
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int, activate_symmetry_breaking: bool = False,
                 use_warm_start: bool = True):
        self.solver = None
        self.n_circuits = n_circuits
        self.board_width = board_width
        self.widths = widths
        self.heights = heights
        self.height_lower_bound = height_lower_bound
        self.height_upper_bound = height_upper_bound
        self.activate_symmetry_breaking = activate_symmetry_breaking
        self.use_warm_start = use_warm_start
        self.time_limit = 300
        self.set_solver("GUROBI_CMD")
        print("Solver set to GUROBI CMD by default. Change solver if needed.")

    @staticmethod
    @abstractmethod
    def from_instance_json(json_filepath: str, activate_symmetry_breaking: bool, use_warm_start: bool) -> "SPulpModel":
        pass

    @staticmethod
    @abstractmethod
    def from_dict(instance_dict: dict, activate_symmetry_breaking: bool, use_warm_start: bool) -> "SPulpModel":
        pass

    def set_time_limit(self, time_limit: int) -> None:
        self.time_limit = time_limit
        print("Time limit set to {}".format(time_limit))

    def set_solver(self, solver_name: str):
        if solver_name not in SOLVERS:
            raise ValueError("Solver %s not supported." % solver_name)
        self.solver = SOLVERS[solver_name](self.time_limit)

    # These are common to all models, so we can implement them here
    def _init_variables(self, *args, **kwargs):
        self.model = pl.LpProblem("S1BM", pl.LpMinimize)

        # Define target variable, the height of the board
        self.board_height = pl.LpVariable("height", lowBound=self.height_lower_bound, upBound=self.height_upper_bound,
                                          cat=LpInteger)

        # Define circuit positioning variables
        self.x = [pl.LpVariable("x_%d" % i, lowBound=0, upBound=self.height_upper_bound - self.widths[i], cat=LpInteger)
                  for i in range(self.n_circuits)]
        self.y = [pl.LpVariable("y_%d" % i, lowBound=self.heights[i], upBound=self.board_width, cat=LpInteger)
                  for i in range(self.n_circuits)]

        # Define auxiliary variables
        self.z_1 = [pl.LpVariable("z_1_%d_%d" % (i, j), lowBound=0, upBound=1, cat=LpInteger) for i in
                    range(self.n_circuits)
                    for j in range(self.n_circuits)]
        self.z_2 = [pl.LpVariable("z_2_%d_%d" % (i, j), lowBound=0, upBound=1, cat=LpInteger) for i in
                    range(self.n_circuits)
                    for j in range(self.n_circuits)]

    def _warm_start_solver(self):
        initial_solution = kp01_upper_bound(np.array(self.widths), np.array(self.heights), self.board_width)
        for i in range(self.n_circuits):
            self.x[i].setInitialValue(initial_solution["y"][i])
            self.y[i].setInitialValue(initial_solution["x"][i] + self.heights[i])

        self.board_height.setInitialValue(initial_solution["board_height"])
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                if self.x[i].varValue < self.x[j].varValue:
                    self.z_1[i * self.n_circuits + j].setInitialValue(1)
                    self.z_1[j * self.n_circuits + i].setInitialValue(0)
                elif self.y[i].varValue < self.y[j].varValue:
                    self.z_2[i * self.n_circuits + j].setInitialValue(0)
                    self.z_2[j * self.n_circuits + i].setInitialValue(1)

        print("Warm start enabled.")

    @abstractmethod
    def _add_model_specific_constraints(self, *args, **kwargs):
        pass

    def _same_sized_circuits_constraints(self):
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                if self.widths[i] == self.widths[j] and self.heights[i] == self.heights[j]:
                    self.model += self.z_1[j * self.n_circuits + i] == 0
                    self.model += self.z_1[i * self.n_circuits + j] + self.z_2[j * self.n_circuits + i] == 1

    def _large_circuits_constraints(self):
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                if self.widths[i] + self.widths[j] > self.board_width:
                    self.model += self.z_1[i * self.n_circuits + j] == 0
                    self.model += self.z_1[j * self.n_circuits + i] == 0
                if self.heights[i] + self.heights[j] > self.height_upper_bound:
                    self.model += self.z_2[i * self.n_circuits + j] == 0
                    self.model += self.z_2[j * self.n_circuits + i] == 0

    def _add_symmetry_breaking_constraint(self):
        areas = [self.widths[i] * self.heights[i] for i in range(self.n_circuits)]
        sorted_indexes = [i for _, i in sorted(zip(areas, range(self.n_circuits)), reverse=True)]

        biggest_circuit_x_upper_bound = (self.height_upper_bound - self.widths[sorted_indexes[0]]) // 2
        biggest_circuit_y_upper_bound = (self.board_width + self.heights[sorted_indexes[0]]) // 2

        self.model += self.x[sorted_indexes[0]] <= biggest_circuit_x_upper_bound
        self.model += self.y[sorted_indexes[0]] <= biggest_circuit_y_upper_bound

        for i in range(self.n_circuits):
            if self.widths[i] > biggest_circuit_x_upper_bound:
                self.model += self.z_1[i * self.n_circuits + sorted_indexes[0]] == 0
            if self.heights[i] > biggest_circuit_y_upper_bound:
                self.model += self.z_2[i * self.n_circuits + sorted_indexes[0]] == 0

    def _define_objective(self):
        self.model += self.board_height

    def _build_model(self):
        self._init_variables()
        self._define_objective()
        self._large_circuits_constraints()
        self._same_sized_circuits_constraints()
        self._add_model_specific_constraints()
        if self.activate_symmetry_breaking:
            self._add_symmetry_breaking_constraint()

        if self.use_warm_start:
            self._warm_start_solver()

    def _retrieve_solution(self):
        xs = []
        ys = []
        for i in range(self.n_circuits):
            xs.append(self.x[i].roundedValue())
            ys.append(self.y[i].roundedValue() - self.heights[i])

        return self.n_circuits, self.widths, self.heights, xs, ys, self.board_height.roundedValue()

    def solve(self):
        start_time = perf_counter()

        print("Building model using Pulp...")
        self._build_model()

        end_time = perf_counter()
        print("It took %.2f seconds to build the model" % (end_time - start_time))

        # Solve the model
        start_time = perf_counter()
        print("Solving the model")
        self.model.solve(self.solver)
        end_time = perf_counter()

        time_limit_exceeded = np.ceil(end_time - start_time) >= self.time_limit

        print("Accessing to the status of the model...")
        print("The status of the model is %s" % pl.LpStatus[self.model.status])

        if self.model.status == 1:
            print("Model solved")
        elif time_limit_exceeded:
            print("Time limit exceeded")
        else:
            print("Model unsatisfiable")

        if time_limit_exceeded:
            return None

        # Get the solution
        return self._retrieve_solution()


class SGBMPulpModel(SPulpModel):
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int, activate_symmetry_breaking: bool = False,
                 use_warm_start: bool = True):
        super().__init__(n_circuits, board_width, widths, heights, height_lower_bound, height_upper_bound,
                         activate_symmetry_breaking, use_warm_start)

    @staticmethod
    def from_instance_json(json_filepath: str, activate_symmetry_breaking: bool = False, use_warm_start: bool = True) \
            -> "SGBMPulpModel":
        with open(json_filepath, 'r') as f:
            instance_dict = json.load(f)

        return SGBMPulpModel(**instance_dict, activate_symmetry_breaking=False, use_warm_start=use_warm_start)

    @staticmethod
    def from_dict(instance_dict: dict, activate_symmetry_breaking: bool = False, use_warm_start: bool = True) \
            -> "SGBMPulpModel":
        return SGBMPulpModel(**instance_dict, activate_symmetry_breaking=False, use_warm_start=use_warm_start)

    def _add_model_specific_constraints(self):
        # Enforce board height to be the maximum y coordinate of any circuit, considering the circuit height
        for i in range(self.n_circuits):
            self.model += self.x[i] + self.widths[i] <= self.board_height

        # Add constraint that circuits do not overlap
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                # i-th circuit placed on the left of j-th circuit
                self.model += self.x[i] + self.widths[i] <= \
                              self.x[j] + (1 - self.z_1[i * self.n_circuits + j]) * self.height_upper_bound
                # i-th circuit placed on the right of j-th circuit
                self.model += self.x[j] + self.widths[j] <= \
                              self.x[i] + (1 - self.z_1[j * self.n_circuits + i]) * self.height_upper_bound
                # i-th circuit placed above j-th circuit
                self.model += self.y[i] - self.heights[i] >= \
                              self.y[j] - (1 - self.z_2[i * self.n_circuits + j]) * self.board_width
                # i-th circuit not on the left of j-th circuit when it is above
                self.model += self.y[j] - self.heights[j] >= \
                              self.y[i] - (1 - self.z_2[j * self.n_circuits + i]) * self.board_width
                # logic proposition only one
                self.model += self.z_1[i * self.n_circuits + j] + self.z_1[j * self.n_circuits + i] + \
                              self.z_2[i * self.n_circuits + j] + self.z_2[j * self.n_circuits + i] == 1


class S1BMPulpModel(SPulpModel):
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int, activate_symmetry_breaking: bool = False,
                 use_warm_start: bool = True):
        super().__init__(n_circuits, board_width, widths, heights, height_lower_bound, height_upper_bound,
                         activate_symmetry_breaking, use_warm_start)


    @staticmethod
    def from_instance_json(json_filepath: str, activate_symmetry_breaking: bool = False, use_warm_start: bool = True) \
            -> "S1BMPulpModel":
        with open(json_filepath, 'r') as f:
            instance_dict = json.load(f)

        return S1BMPulpModel(**instance_dict, activate_symmetry_breaking=False, use_warm_start=use_warm_start)

    @staticmethod
    def from_dict(instance_dict: dict, activate_symmetry_breaking: bool = False, use_warm_start: bool = True) \
            -> "S1BMPulpModel":
        return S1BMPulpModel(**instance_dict, activate_symmetry_breaking=False, use_warm_start=use_warm_start)

    def _add_model_specific_constraints(self):
        # Enforce board height to be the maximum y coordinate of any circuit, considering the circuit height
        for i in range(self.n_circuits):
            self.model += self.x[i] + self.widths[i] <= self.board_height

        # Add constraint that circuits do not overlap
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                # i-th circuit placed on the left of j-th circuit
                self.model += self.x[i] + self.widths[i] <= \
                              self.x[j] + (1 - self.z_1[i * self.n_circuits + j]) * self.height_upper_bound
                # i-th circuit placed on the right of j-th circuit
                self.model += self.x[j] + self.widths[j] <= \
                              self.x[i] + (1 - self.z_1[j * self.n_circuits + i]) * self.height_upper_bound
                # i-th circuit placed above j-th circuit
                self.model += self.y[i] - self.heights[i] >= \
                              self.y[j] - (1 - self.z_2[i * self.n_circuits + j]) * self.board_width
                self.model += self.x[i] + self.widths[i] >= \
                              self.x[j] - (1 - self.z_2[i * self.n_circuits + j]) * self.height_upper_bound
                self.model += self.x[j] + self.widths[j] >= \
                              self.x[i] - (1 - self.z_2[i * self.n_circuits + j]) * self.height_upper_bound
                # i-th circuit not on the left of j-th circuit when it is above
                self.model += self.y[j] - self.heights[j] >= \
                              self.y[i] - (1 - self.z_2[j * self.n_circuits + i]) * self.board_width
                self.model += self.x[i] + self.widths[i] >= \
                              self.x[j] - (1 - self.z_2[j * self.n_circuits + i]) * self.height_upper_bound
                self.model += self.x[j] + self.widths[j] >= \
                              self.x[i] - (1 - self.z_2[j * self.n_circuits + i]) * self.height_upper_bound
                # logic proposition only one
                self.model += self.z_1[i * self.n_circuits + j] + self.z_1[j * self.n_circuits + i] + \
                              self.z_2[i * self.n_circuits + j] + self.z_2[j * self.n_circuits + i] == 1


class S2BMPulpModel(SPulpModel):
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int, activate_symmetry_breaking: bool = False,
                 use_warm_start: bool = True):
        super().__init__(n_circuits, board_width, widths, heights, height_lower_bound, height_upper_bound,
                         activate_symmetry_breaking, use_warm_start)

    @staticmethod
    def from_instance_json(json_filepath: str, activate_symmetry_breaking: bool = False, use_warm_start: bool = True) \
            -> "S2BMPulpModel":
        with open(json_filepath, 'r') as f:
            instance_dict = json.load(f)

        return S2BMPulpModel(**instance_dict, activate_symmetry_breaking=False, use_warm_start=use_warm_start)

    @staticmethod
    def from_dict(instance_dict: dict, activate_symmetry_breaking: bool = False, use_warm_start: bool = True) -> "S2BMPulpModel":
        return S2BMPulpModel(**instance_dict, activate_symmetry_breaking=False, use_warm_start=use_warm_start)


    def _add_model_specific_constraints(self):
        # Enforce board height to be the maximum y coordinate of any circuit, considering the circuit height
        for i in range(self.n_circuits):
            self.model += self.x[i] + self.widths[i] <= self.board_height

        # Add constraint that circuits do not overlap
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                # i-th circuit placed on the left of j-th circuit
                self.model += self.x[i] + self.widths[i] <= \
                              self.x[j] + (1 - self.z_1[i * self.n_circuits + j]) * self.height_upper_bound
                # i-th circuit placed on the right of j-th circuit
                self.model += self.x[j] + self.widths[j] <= \
                              self.x[i] + (1 - self.z_1[j * self.n_circuits + i]) * self.height_upper_bound
                # i-th circuit placed above j-th circuit
                self.model += self.y[i] - self.heights[i] >= \
                              self.y[j] - (1 - self.z_2[i * self.n_circuits + j]) * self.board_width
                self.model += self.x[i] + self.widths[i] >= \
                              self.x[j] + 1 - (1 - self.z_2[i * self.n_circuits + j]) * self.height_upper_bound
                self.model += self.x[j] + self.widths[j] >= \
                              self.x[i] + 1 - (1 - self.z_2[i * self.n_circuits + j]) * self.height_upper_bound
                # i-th circuit not on the left of j-th circuit when it is above
                self.model += self.y[j] - self.heights[j] >= \
                              self.y[i] - (1 - self.z_2[j * self.n_circuits + i]) * self.board_width
                self.model += self.x[i] + self.widths[i] >= \
                              self.x[j] + 1 - (1 - self.z_2[j * self.n_circuits + i]) * self.height_upper_bound
                self.model += self.x[j] + self.widths[j] >= \
                              self.x[i] + 1 - (1 - self.z_2[j * self.n_circuits + i]) * self.height_upper_bound
                # logic proposition only one
                self.model += self.z_1[i * self.n_circuits + j] + self.z_1[j * self.n_circuits + i] + \
                              self.z_2[i * self.n_circuits + j] + self.z_2[j * self.n_circuits + i] == 1
