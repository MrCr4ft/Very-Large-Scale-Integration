import time
import typing
import json
import subprocess


def spawn_z3():
    return subprocess.Popen(
        "z3 -in",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )


def spawn_cvc5():
    return subprocess.Popen(
        "cvc5",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )


def lexicographic_ordering_ror(aux_vars_prefix: str, vector_a: typing.List[str], vector_b: typing.List[str]) -> str:
    aux_vars = [aux_vars_prefix + str(i) for i in range(len(vector_a))]
    aux_vars_definition = ""
    n = len(vector_a)
    for i in range(n):
        aux_vars_definition += "(declare-fun {}{} () Bool)\n".format(aux_vars_prefix, i)

    constraints = "(assert {})\n".format(aux_vars[0])
    constraints += "(assert " + \
                   iff("{}".format(aux_vars[n - 1]), "(<= {} {})".format(vector_a[n - 1], vector_b[n - 1])) + \
                   ")\n"

    for i in range(1, n):
        constraints += "(assert " + \
                       iff(aux_vars[n - i - 1], "(or (< {} {}) (and (= {} {}) {}))".format(
                           vector_a[n - i - 1], vector_b[n - i - 1],
                           vector_a[n - i - 1], vector_b[n - i - 1],
                           aux_vars[n - i])) + ")\n"

    return aux_vars_definition + "\n\n" + constraints


def iff(a: str, b: str) -> str:
    return "(and (=> {} {}) (=> {} {}))".format(a, b, b, a)


def retrieve_value(solver: subprocess.Popen, var_name: str) -> int:
    solver.stdin.write("(get-value ({}))\n".format(var_name).encode("utf-8"))
    solver.stdin.flush()
    result = solver.stdout.readline().decode("utf-8")
    return int(result.split(" ")[-1].replace(")", ""))


class SMTModel:
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int, time_limit_ms: int, allow_rotation: bool = False):
        self.n_circuits = n_circuits
        self.board_width = board_width
        self.widths = widths
        self.heights = heights
        self.height_lower_bound = height_lower_bound
        self.height_upper_bound = height_upper_bound
        self.time_limit_ms = time_limit_ms
        self.allow_rotation = allow_rotation
        print("Time limit set to: {}".format(self.time_limit_ms))

    @staticmethod
    def from_instance_json(json_filepath: str, allow_rotation: bool, time_limit_ms: int,
                           *args, **kwargs) \
            -> "SMTModel":
        with open(json_filepath, "r") as f:
            instance = json.load(f)

        return SMTModel(**instance, time_limit_ms=time_limit_ms, allow_rotation=allow_rotation)

    def _get_smt_lib_options(self) -> str:
        return "(set-option :produce-models true)\n(set-logic QF_LIA)\n". \
            format(self.time_limit_ms)

    def set_time_limit(self, time_limit_ms: int):
        self.time_limit_ms = time_limit_ms

    def _declare_smt_lib_variables(self) -> str:
        variables_declaration = "(declare-fun board_height () Int)\n"
        for i in range(self.n_circuits):
            variables_declaration += "(declare-fun x{} () Int)\n(declare-fun y{} () Int)\n".format(i, i)

        if self.allow_rotation:
            for i in range(self.n_circuits):
                variables_declaration += "(declare-fun r{} () Bool)\n".format(i)
                variables_declaration += "(declare-fun rw{} () Int)\n(declare-fun rh{} () Int)\n".format(i, i)

        return variables_declaration

    def _enforce_rotation(self) -> str:
        rotation_constraints = ""
        for i in range(self.n_circuits):
            rotation_constraints += "(assert (ite r{} (= rw{} {}) (= rw{} {})))\n".format(i, i, self.heights[i],
                                                                                          i, self.widths[i])
            rotation_constraints += "(assert (ite r{} (= rh{} {}) (= rh{} {})))\n".format(i, i, self.widths[i],
                                                                                          i, self.heights[i])

        return rotation_constraints

    def _impose_variables_domain(self) -> str:
        variables_domain_assertions = ""
        for i in range(self.n_circuits):
            variables_domain_assertions += "(assert (>= x{} 0))\n".format(i)
            variables_domain_assertions += "(assert (>= y{} 0))\n".format(i)

            if self.allow_rotation:
                variables_domain_assertions += "(assert (<= x{} (- {} rw{})))\n".format(i, self.board_width, i)
                variables_domain_assertions += "(assert (<= y{} (- board_height rh{})))\n".format(i, i)
            else:
                variables_domain_assertions += "(assert (<= x{} {}))\n".format(i, self.board_width - self.widths[i])
                variables_domain_assertions += "(assert (<= y{} {}))\n".format(i,
                                                                               self.height_upper_bound - self.heights[
                                                                                   i])

        return variables_domain_assertions

    def _non_overlapping_constraint(self, i: int, j: int) -> str:
        # x[i] + dx[i] <= x[j] \/ y[i] + dy[i] <= y[j] \/
        # x[j] + dx[j] <= x[i] \/ y[j] + dy[j] <= y[i]
        if self.allow_rotation:
            return "(assert (or (<= (+ x{} rw{}) x{}) (<= (+ y{} rh{}) y{}) " \
                   "(<= (+ x{} rw{}) x{}) (<= (+ y{} rh{}) y{})))\n".format(i, i, j, i, i, j,
                                                                            j, j, i, j, j, i)
        else:
            return "(assert (or (<= (+ x{} {}) x{}) (<= (+ y{} {}) y{}) " \
                   "(<= (+ x{} {}) x{}) (<= (+ y{} {}) y{})))\n".format(i, self.widths[i], j, i, self.heights[i], j,
                                                                        j, self.widths[j], i, j, self.heights[j], i)

    def cumulative_constraints_x(self):
        constraints = ""
        early = 0
        late = self.board_width + max(self.widths)
        for t in range(early, late + 1):
            sum_expr = "(+ "
            for i in range(self.n_circuits):
                if self.allow_rotation:
                    bool_expr = "(and (<= x{} {}) (< {} (+ x{} rw{})))".format(i, t, t, i, i)
                else:
                    bool_expr = "(and (<= x{} {}) (< {} (+ x{} {})))".format(i, t, t, i, self.widths[i])
                bool_to_int_expr = "(ite {} 1 0)".format(bool_expr)
                if self.allow_rotation:
                    sum_expr += "(* {} rh{}) ".format(bool_to_int_expr, i)
                else:
                    sum_expr += "(* {} {}) ".format(bool_to_int_expr, self.heights[i])
            sum_expr += ")"
            constraints += "(assert (<= {} board_height))\n".format(sum_expr)

        return constraints

    def cumulative_constraints_y(self):
        constraints = ""
        early = 0
        late = self.height_upper_bound + max(self.heights)
        for t in range(early, late + 1):
            sum_expr = "(+ "
            for i in range(self.n_circuits):
                if self.allow_rotation:
                    bool_expr = "(and (<= y{} {}) (< {} (+ y{} rh{})))".format(i, t, t, i, i)
                else:
                    bool_expr = "(and (<= y{} {}) (< {} (+ y{} {})))".format(i, t, t, i, self.heights[i])
                bool_to_int_expr = "(ite {} 1 0)".format(bool_expr)
                if self.allow_rotation:
                    sum_expr += "(* {} rw{}) ".format(bool_to_int_expr, i)
                else:
                    sum_expr += "(* {} {}) ".format(bool_to_int_expr, self.widths[i])
            sum_expr += ")"

            constraints += "(assert (<= {} {}))\n".format(sum_expr, self.board_width)

        return constraints

    def _large_circuits_pair_no_rot(self):
        constraints = ""
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                if self.widths[i] + self.widths[j] > self.board_width:
                    constraints += "(assert (or (<= (+ y{i} {height_i}) y{j}) (>= y{i} (+ y{j} {height_j}))))\n". \
                        format(i=i, j=j, height_i=self.heights[i], height_j=self.heights[j])
                if self.heights[i] + self.heights[j] > self.height_upper_bound:
                    constraints += "(assert (=> (>= {cumulative_height} board_height) " \
                                   "(or (<= (+ x{i} {width_i}) x{j}) (>= x{i} (+ x{j} {width_j})))))\n".\
                        format(cumulative_height=self.heights[i] + self.heights[j], i=i, j=j, width_i=self.widths[i],
                               width_j=self.widths[j])
        return constraints

    def lexicographic_ordering_symmetry_breaking(self):
        constraints = ""
        for i in range(self.n_circuits):
            if not self.allow_rotation:
                reflected_x_positions = ["(- {} {} x{})".format(self.board_width, self.widths[i], i) for i
                                         in
                                         range(self.n_circuits)]
                reflected_y_positions = ["(- board_height {} y{})".format(self.heights[i], i)
                                         for i in range(self.n_circuits)]
                constraints += lexicographic_ordering_ror("aux_lex_x_{}_".format(i),
                                                          ["x{}".format(i) for i in range(self.n_circuits)],
                                                          reflected_x_positions)
                constraints += lexicographic_ordering_ror("aux_lex_y_{}_".format(i),
                                                          ["y{}".format(i) for i in range(self.n_circuits)],
                                                          reflected_y_positions)
            else:
                reflected_x_positions = ["(- {} rw{} x{})".format(self.board_width, i, i) for i
                                         in range(self.n_circuits)]
                reflected_y_positions = ["(- board_height rh{} y{})".format(i, i)
                                         for i in range(self.n_circuits)]
                constraints += lexicographic_ordering_ror("aux_lex_x_{}_".format(i),
                                                          ["x{}".format(i) for i in range(self.n_circuits)],
                                                          reflected_x_positions)
                constraints += lexicographic_ordering_ror("aux_lex_y_{}_".format(i),
                                                          ["y{}".format(i) for i in range(self.n_circuits)],
                                                          reflected_y_positions)

        return constraints

    def _enforce_actual_board_height(self):
        constraints = ""
        for i in range(self.n_circuits):
            if not self.allow_rotation:
                constraints += "(assert (<= (+ y{} {}) board_height))\n".format(i, self.heights[i])
            else:
                constraints += "(assert (<= (+ y{} rh{}) board_height))\n".format(i, i)
        return constraints

    def one_pair_of_circuits_symmetry_breaking(self):
        areas = [(i, self.widths[i] * self.heights[i]) for i in range(self.n_circuits)]
        areas.sort(key=lambda x: x[1], reverse=True)
        o1 = areas[1][0]
        o2 = areas[2][0]

        return lexicographic_ordering_ror("aux_lex_sb_", ["y{}".format(o1), "x{}".format(o1)],
                                          ["y{}".format(o2), "x{}".format(o2)])

    def to_smt_lib_format(self, turn_on_cumulative_constraints: bool = False,
                          turn_on_symmetry_breaking: bool = True) -> str:
        smt_lib_instance = self._get_smt_lib_options() + "\n" + self._declare_smt_lib_variables() + "\n" + \
                           self._impose_variables_domain() + "\n"

        smt_lib_instance += self._enforce_actual_board_height()

        if self.allow_rotation:
            smt_lib_instance += self._enforce_rotation() + "\n"

        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                smt_lib_instance += self._non_overlapping_constraint(i, j)

        if not self.allow_rotation:
            smt_lib_instance += self._large_circuits_pair_no_rot()

        if turn_on_cumulative_constraints:
            smt_lib_instance += self.cumulative_constraints_x()
            smt_lib_instance += self.cumulative_constraints_y()

        if turn_on_symmetry_breaking:
            smt_lib_instance += self.lexicographic_ordering_symmetry_breaking()
            smt_lib_instance += self.one_pair_of_circuits_symmetry_breaking()

        return smt_lib_instance

    def retrieve_solution(self, solver: subprocess.Popen) -> typing.Dict[str, typing.Any]:
        xs = []
        ys = []
        ws = []
        hs = []
        for i in range(self.n_circuits):
            x = retrieve_value(solver, "x{}".format(i))
            xs.append(x)
            y = retrieve_value(solver, "y{}".format(i))
            ys.append(y)
            if self.allow_rotation:
                w = retrieve_value(solver, "rw{}".format(i))
                h = retrieve_value(solver, "rh{}".format(i))
                ws.append(w)
                hs.append(h)
        board_height = retrieve_value(solver, "board_height")

        return {
            'board_width': self.board_width,
            'board_height': board_height,
            'n_circuits': self.n_circuits,
            'widths': ws if self.allow_rotation else self.widths,
            'heights': hs if self.allow_rotation else self.heights,
            'x': xs,
            'y': ys
        }

    def solve(self, turn_on_cumulative_constraints: bool = False, turn_on_symmetry_breaking: bool = True,
              solver: str = "z3", *args, **kwargs):
        smt_lib_model = self.to_smt_lib_format(turn_on_cumulative_constraints, turn_on_symmetry_breaking)
        if solver == "z3":
            solver = spawn_z3()
        elif solver == "cvc5":
            solver = spawn_cvc5()
        else:
            raise ValueError("Solver: {} not configured!".format(solver))
        solver.stdin.write(smt_lib_model.encode("utf-8"))

        # binary searchf
        lb = self.height_lower_bound
        ub = self.height_upper_bound
        current_time_limit = self.time_limit_ms
        solution = None
        while lb <= ub:
            mid = (lb + ub) // 2
            print("Trying to solve with height equal to ", mid)
            start_time = time.perf_counter()
            solver.stdin.write("(push)".format(current_time_limit).encode("utf-8"))
            solver.stdin.write("(set-option :timeout {})".format(self.time_limit_ms).encode("utf-8"))
            solver.stdin.write("(assert (<= board_height {}))\n".format(mid).encode("utf-8"))
            solver.stdin.write("(check-sat)\n".encode("utf-8"))
            solver.stdin.flush()
            solver_output = solver.stdout.readline().decode("utf-8").strip()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            current_time_limit -= int(elapsed_time * 1000)
            if current_time_limit <= 0:
                print("Time limit exceeded")
                if solution is not None:
                    return solution, self.time_limit_ms, False
                else:
                    solver.terminate()
                    return None, self.time_limit_ms, False
            if solver_output == "sat":
                ub = mid - 1
                print("Solution found with height equal to {}!".format(mid))
                solution = self.retrieve_solution(solver)
            elif solver_output == "unsat":
                lb = mid + 1
                print("Unsat with height equal to {}!".format(mid))
                solver.stdin.write("(pop)".format(current_time_limit).encode("utf-8"))
            else:
                raise Exception("Unknown solver output: {}. Exiting...".format(solver_output))

        solver.terminate()
        return solution, self.time_limit_ms - current_time_limit, True
