import typing
import json


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


class Model:
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, height_upper_bound: int, time_limit_ms: int):
        self.n_circuits = n_circuits
        self.board_width = board_width
        self.widths = widths
        self.heights = heights
        self.height_lower_bound = height_lower_bound
        self.height_upper_bound = height_upper_bound
        self.time_limit_ms = time_limit_ms
        print("Time limit set to: {}".format(self.time_limit_ms))

    @staticmethod
    def from_instance_json(json_filepath: str, time_limit_ms: int) -> "Model":
        with open(json_filepath, "r") as f:
            instance = json.load(f)

        return Model(**instance, time_limit_ms=time_limit_ms)

    def _get_smt_lib_options(self) -> str:
        return "(set-option :produce-models true)\n(set-option :timeout {})\n(set-logic QF_LIA)\n". \
            format(self.time_limit_ms)

    def _declare_smt_lib_variables(self) -> str:
        variables_declaration = "(declare-fun board_height () Int)\n"
        for i in range(self.n_circuits):
            variables_declaration += "(declare-fun x{} () Int)\n(declare-fun y{} () Int)\n".format(i, i)

        return variables_declaration

    def _impose_variables_domain(self) -> str:
        variables_domain_assertions = ""
        for i in range(self.n_circuits):
            variables_domain_assertions += "(assert (>= x{} 0))\n" \
                                           "(assert (>= y{} 0))\n" \
                                           "(assert (<= x{} (- {} {})))\n" \
                                           "(assert (<= y{} (- {} {})))\n".format(i, i, i, self.board_width,
                                                                                  self.widths[i], i,
                                                                                  self.height_upper_bound,
                                                                                  self.heights[i])

        return variables_domain_assertions

    def _impose_effective_board_height(self) -> str:
        effective_board_height_constraint = ""
        for i in range(self.n_circuits):
            effective_board_height_constraint += "(assert (<= y{} (- board_height {})))\n".format(i, self.heights[i])

        return effective_board_height_constraint

    def _non_overlapping_constraint(self, i: int, j: int) -> str:
        # x[i] + dx[i] <= x[j] \/ y[i] + dy[i] <= y[j] \/
        # x[j] + dx[j] <= x[i] \/ y[j] + dy[j] <= y[i]
        return "(assert (or (<= (+ x{} {}) x{}) (<= (+ y{} {}) y{}) " \
               "(<= (+ x{} {}) x{}) (<= (+ y{} {}) y{})))\n".format(i, self.widths[i], j, i, self.heights[i], j,
                                                                    j, self.widths[j], i, j, self.heights[j], i)

    def _allowable_circuits_positions(self, vertical: bool = False) -> str:
        allowable_circuits_positions_constraints = ""
        variable = "x" if not vertical else "y"
        for i in range(self.n_circuits):
            allowable_circuits_positions_constraints += "(assert (or "
            allowable_circuits_positions_constraints += "(= {}{} 0) ".format(variable, i)
            for j in range(self.n_circuits):
                if i != j:
                    allowable_circuits_positions_constraints += "(= (+ {}{} {}) {}{}) ". \
                        format(variable, j, self.widths[j], variable, i)

            allowable_circuits_positions_constraints += "))\n"

        return allowable_circuits_positions_constraints

    def cumulative_constraints_x(self):
        constraints = ""
        early = 0
        late = self.board_width + max(self.widths)
        for t in range(early, late + 1):
            sum_expr = "(+ "
            for i in range(self.n_circuits):
                bool_expr = "(and (<= x{} {}) (< {} (+ x{} {})))".format(i, t, t, i, self.widths[i])
                bool_to_int_expr = "(ite {} 1 0)".format(bool_expr)
                sum_expr += "(* {} {}) ".format(bool_to_int_expr, self.heights[i])
            sum_expr += ")"

            constraints += "(assert (<= {} {}))\n".format(sum_expr, self.height_upper_bound)

        return constraints

    def cumulative_constraints_y(self):
        constraints = ""
        early = 0
        late = self.height_upper_bound + max(self.heights)
        for t in range(early, late + 1):
            sum_expr = "(+ "
            for i in range(self.n_circuits):
                bool_expr = "(and (<= y{} {}) (< {} (+ y{} {})))".format(i, t, t, i, self.heights[i])
                bool_to_int_expr = "(ite {} 1 0)".format(bool_expr)
                sum_expr += "(* {} {}) ".format(bool_to_int_expr, self.widths[i])
            sum_expr += ")"

            constraints += "(assert (<= {} {}))\n".format(sum_expr, self.board_width)

        return constraints

    def _large_circuits_pair(self):
        constraints = ""
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                if self.widths[i] + self.widths[j] > self.board_width:
                    constraints += "(assert (or (<= (+ y{} {}) y{}) (>= y{} (+ y{} {}))))\n". \
                        format(i, self.heights[i], j, i, j, self.heights[j])
                if self.heights[i] + self.heights[j] > self.height_upper_bound:
                    constraints += "(assert (or (<= (+ x{} {}) x{}) (>= x{} (+ x{} {}))))\n". \
                        format(i, self.widths[i], j, i, j, self.widths[j])

        return constraints

    def lexicographic_ordering_symmetry_breaking(self):
        constraints = ""
        for i in range(self.n_circuits):
            reflected_x_positions = ["(- {} {} {})".format(self.board_width, self.widths[i], "x{}".format(i)) for i in
                                     range(self.n_circuits)]
            reflected_y_positions = ["(- {} {} {})".format(self.height_upper_bound, self.heights[i], "y{}".format(i))
                                     for i in range(self.n_circuits)]
            constraints += lexicographic_ordering_ror("aux_lex_x_{}_".format(i),
                                                      ["x{}".format(i) for i in range(self.n_circuits)],
                                                      reflected_x_positions)
            constraints += lexicographic_ordering_ror("aux_lex_y_{}_".format(i),
                                                      ["y{}".format(i) for i in range(self.n_circuits)],
                                                      reflected_y_positions)

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

        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                smt_lib_instance += self._non_overlapping_constraint(i, j)

        smt_lib_instance += self._large_circuits_pair()

        if turn_on_cumulative_constraints:
            smt_lib_instance += self.cumulative_constraints_x()
            smt_lib_instance += self.cumulative_constraints_y()

        if turn_on_symmetry_breaking:
            smt_lib_instance += self.lexicographic_ordering_symmetry_breaking()
            smt_lib_instance += self.one_pair_of_circuits_symmetry_breaking()

        smt_lib_instance += self._impose_effective_board_height()
        smt_lib_instance += "(assert (= board_height {}))\n".format(self.height_lower_bound)
        smt_lib_instance += "(check-sat)\n(get-model)\n"

        return smt_lib_instance
