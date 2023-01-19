import typing
import json
from time import perf_counter

from z3 import *
import numpy as np


class OrderEncodedVariable:
    def __init__(self, name: str, lower_bound: int, upper_bound: int):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.order_encoding_booleans = [
            Bool(f"p{self.name}_{domain_val}") for domain_val in range(self.lower_bound - 1, self.upper_bound + 1)
        ]

    def __str__(self):
        return self.name + "[" + str(self.lower_bound) + ", " + str(self.upper_bound) + "]"

    def actual_value(self, model: z3.ModelRef):
        model_eval_order_encoding_booleans = [is_true(model[self.order_encoding_booleans[domain_val]]) for domain_val in
                                              range(self.upper_bound - self.lower_bound + 2)]

        return self.lower_bound - 1 + np.argmax(model_eval_order_encoding_booleans)

    def get_axiom_clauses(self):
        axiom_clauses_ = list()
        axiom_clauses_.append(Not(self.order_encoding_booleans[0]))
        axiom_clauses_.append(self.order_encoding_booleans[self.upper_bound - self.lower_bound + 1])
        for c in range(1, self.upper_bound - self.lower_bound + 2):
            axiom_clauses_.append(Or(Not(self.order_encoding_booleans[c - 1]), self.order_encoding_booleans[c]))

        return axiom_clauses_


class SATStripPackingModel:
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int, activate_symmetry_breaking: bool = False,
                 add_implied_constraints: bool = False):
        self.y_vars = None
        self.x_vars = None
        self.board_height_actual_value = None
        self.board_height = OrderEncodedVariable("board_height", height_lower_bound, height_upper_bound)
        self.n_circuits = n_circuits
        self.board_width = board_width
        self.widths = widths
        self.heights = heights
        self.height_lower_bound = height_lower_bound
        self.height_upper_bound = height_upper_bound
        self.activate_symmetry_breaking = activate_symmetry_breaking
        self.add_implied_constraints = add_implied_constraints
        self.time_limit = 300
        print("Time limit set to: {}".format(self.time_limit))

    @staticmethod
    def from_instance_json(json_filepath: str, activate_symmetry_breaking: bool = False,
                           add_implied_constraints: bool = False):
        with open(json_filepath, "r") as f:
            instance = json.load(f)

        return SATStripPackingModel(**instance, activate_symmetry_breaking=activate_symmetry_breaking,
                                    add_implied_constraints=add_implied_constraints)

    @staticmethod
    def from_dict(instance_dict: dict, activate_symmetry_breaking: bool = False,
                  add_implied_constraints: bool = False):
        return SATStripPackingModel(**instance_dict, activate_symmetry_breaking=activate_symmetry_breaking,
                                    add_implied_constraints=add_implied_constraints)

    def set_time_limit(self, time_limit: int) -> None:
        self.time_limit = time_limit
        print("Time limit set to {}".format(time_limit))

    def _init_z3_variables(self):
        x_lower_bounds = [0] * self.n_circuits
        x_upper_bounds = [self.board_width - self.widths[i] for i in range(self.n_circuits)]

        y_lower_bounds = [0] * self.n_circuits
        y_upper_bounds = [self.height_upper_bound - self.heights[i] for i in range(self.n_circuits)]

        self.x_vars = [OrderEncodedVariable(f"x_{i + 1}", x_lower_bounds[i], x_upper_bounds[i])
                       for i in range(self.n_circuits)]
        self.y_vars = [OrderEncodedVariable(f"y_{i + 1}", y_lower_bounds[i], y_upper_bounds[i])
                       for i in range(self.n_circuits)]

        self.lr = [[Bool(f"lr_{i + 1}_{j + 1}") for j in range(self.n_circuits)] for i in range(self.n_circuits)]
        self.ud = [[Bool(f"ud_{i + 1}_{j + 1}") for j in range(self.n_circuits)] for i in range(self.n_circuits)]

    def _basic_constraints(self):
        basic_constraints = list()

        # 1. x_vars and y_vars Axiom Clauses
        for i in range(self.n_circuits):
            basic_constraints += self.x_vars[i].get_axiom_clauses()
            basic_constraints += self.y_vars[i].get_axiom_clauses()

        # 2. Board height Axiom Clauses
        basic_constraints += self.board_height.get_axiom_clauses()

        # 3. Board height constraints
        basic_constraints += self._board_height_constraints()

        # 4. No overlap
        basic_constraints += self._no_overlapping_constraints()

        # 5. Exclusive placing relationship implied constraints
        if self.add_implied_constraints:
            basic_constraints += self._exclusive_constraints()

        # 6. Same sized circuits symmetry breaking
        if self.activate_symmetry_breaking:
            basic_constraints += self._same_sized_circuits_symmetry_breaking_constraints()

        return basic_constraints

    def _no_overlapping_constraints(self):
        constraints = list()

        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                for domain_var in range(-1, self.board_width - self.widths[i]):
                    if domain_var + self.widths[i] <= self.board_width - self.widths[j]:
                        constraints.append(
                            Or(
                                Not(self.lr[i][j]), self.x_vars[i].order_encoding_booleans[domain_var + 1],
                                Not(self.x_vars[j].order_encoding_booleans[domain_var + 1 + self.widths[i]])
                            )
                        )
                for domain_var in range(-1, self.board_width - self.widths[j]):
                    if domain_var + self.widths[j] <= self.board_width - self.widths[i]:
                        constraints.append(
                            Or(
                                Not(self.lr[j][i]), self.x_vars[j].order_encoding_booleans[domain_var + 1],
                                Not(self.x_vars[i].order_encoding_booleans[domain_var + 1 + self.widths[j]])
                            )
                        )

                for domain_var in range(-1, self.height_upper_bound - self.heights[i]):
                    if domain_var + self.heights[i] <= self.height_upper_bound - self.heights[j]:
                        constraints.append(
                            Or(
                                Not(self.ud[i][j]), self.y_vars[i].order_encoding_booleans[domain_var + 1],
                                Not(self.y_vars[j].order_encoding_booleans[domain_var + 1 + self.heights[i]])
                            )
                        )
                for domain_var in range(-1, self.height_upper_bound - self.heights[j]):
                    if domain_var + self.heights[j] <= self.height_upper_bound - self.heights[i]:
                        constraints.append(
                            Or(
                                Not(self.ud[j][i]), self.y_vars[j].order_encoding_booleans[domain_var + 1],
                                Not(self.y_vars[i].order_encoding_booleans[domain_var + 1 + self.heights[j]])
                            )
                        )

                constraints.append(Or(self.lr[i][j], self.lr[j][i], self.ud[i][j], self.ud[j][i]))

        return constraints

    def _board_height_constraints(self):
        constraints = list()
        for i in range(self.n_circuits):
            for o in range(self.height_lower_bound, self.height_upper_bound):
                constraints.append(
                    Or(
                        Not(self.board_height.order_encoding_booleans[o - self.height_lower_bound + 1]),
                        self.y_vars[i].order_encoding_booleans[o - self.heights[i] + 1]
                    )
                )

        return constraints

    def _exclusive_constraints(self):
        constraints = list()
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                constraints.append(Or(Not(self.lr[i][j]), Not(self.lr[j][i])))
                constraints.append(Or(Not(self.ud[i][j]), Not(self.ud[j][i])))

        return constraints

    def _large_circuits_constraints(self):
        constraints = list()
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                if self.widths[i] + self.widths[j] > self.board_width:
                    constraints.append(Not(self.lr[i][j]))
                    constraints.append(Not(self.lr[j][i]))
                if self.heights[i] + self.heights[j] > self.board_height_actual_value:
                    constraints.append(Not(self.ud[i][j]))
                    constraints.append(Not(self.ud[j][i]))

        return constraints

    def _same_sized_circuits_symmetry_breaking_constraints(self):
        constraints = list()
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                if self.widths[i] == self.widths[j] and self.heights[i] == self.heights[j]:
                    constraints.append(Not(self.lr[j][i]))
                    constraints.append(Or(self.lr[i][j], Not(self.ud[j][i])))

        return constraints

    def _domain_reduction_symmetry_breaking_constraints(self):
        constraints = []

        d = np.argmax(self.widths)
        x_d_upper_bound = math.floor((self.board_width - self.widths[d]) / 2)
        y_d_upper_bound = math.floor((self.board_height_actual_value - self.heights[d]) / 2)

        for e in range(x_d_upper_bound, self.board_width - self.widths[d]):
            constraints.append(self.x_vars[d].order_encoding_booleans[e + 1])
        for f in range(y_d_upper_bound, self.board_height_actual_value - self.heights[d]):
            constraints.append(self.y_vars[d].order_encoding_booleans[f + 1])

        for i in range(self.n_circuits):
            if self.widths[i] > x_d_upper_bound:
                constraints.append(Not(self.lr[i][d]))
            if self.heights[i] > y_d_upper_bound:
                constraints.append(Not(self.ud[i][d]))

        return constraints

    def _set_board_height_actual_value(self, board_height: int):
        self.board_height_actual_value = board_height
        return self.board_height.order_encoding_booleans[board_height - self.height_lower_bound + 1]

    def _retrieve_solution(self, model):
        xs = []
        ys = []
        for i in range(self.n_circuits):
            xs.append(self.x_vars[i].actual_value(model))
            ys.append(self.y_vars[i].actual_value(model))

        return {
            'board_width': self.board_width,
            'board_height': self.board_height_actual_value,
            'n_circuits': self.n_circuits,
            'widths': self.widths,
            'heights': self.heights,
            'x': xs,
            'y': ys
        }

    def solve(self, linear_search: bool = True):
        self._init_z3_variables()
        s = Solver()
        s.set("timeout", self.time_limit * 1000)

        print("Adding basic constraints to the solver...")
        s.add(self._basic_constraints())
        model = None

        start_time = perf_counter()

        if linear_search:
            lb = self.height_lower_bound
            ub = self.height_upper_bound
            while lb <= ub:
                print("Trying to solve with board height equal to {}".format(lb))
                s.push()
                s.add(self._set_board_height_actual_value(lb))
                if self.add_implied_constraints:
                    s.add(self._large_circuits_constraints())
                if self.activate_symmetry_breaking:
                    s.add(self._domain_reduction_symmetry_breaking_constraints())
                evaluation = s.check()
                if evaluation == sat:
                    print("Found solution with board height equal to {}".format(lb))
                    model = s.model()
                    lb = ub + 1
                elif evaluation == unknown:
                    print("Reached timeout. Terminating...")
                    model = None
                    break
                elif evaluation == unsat:
                    print("Unsatisfiable with board height equal to {}".format(lb))
                    lb = lb + 1
                else:
                    raise Exception("Unknown evaluation: {}".format(evaluation))
                s.pop()

        else:
            lb = self.height_lower_bound
            ub = self.height_upper_bound
            while lb <= ub:
                mid = (lb + ub) // 2
                print("Trying to solve with board height equal to {}".format(mid))
                s.push()
                s.add(self._set_board_height_actual_value(mid))
                if self.add_implied_constraints:
                    s.add(self._large_circuits_constraints())
                if self.activate_symmetry_breaking:
                    s.add(self._domain_reduction_symmetry_breaking_constraints())

                evaluation = s.check()
                if evaluation == sat:
                    print("Found solution with board height equal to {}".format(mid))
                    model = s.model()
                    ub = mid - 1
                elif evaluation == unknown:
                    print("Reached timeout. Terminating...")
                    model = None
                    break
                elif evaluation == unsat:
                    print("Unsatisfiable with board height equal to {}".format(mid))
                    lb = mid + 1
                else:
                    raise Exception("Unknown evaluation: {}".format(evaluation))
                s.pop()

        end_time = perf_counter()
        elapsed_time = np.ceil(end_time - start_time)
        print("Total time elapsed (in seconds): {}".format(elapsed_time))

        time_limit_exceeded = elapsed_time >= self.time_limit

        if model is not None and not time_limit_exceeded:
            return self._retrieve_solution(model)

        return None
