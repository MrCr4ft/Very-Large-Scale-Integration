import typing
import json


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
        return "(set-option :produce-models true)\n(set-option :timeout {})\n(set-logic QF_LIA)\n".\
            format(self.time_limit_ms)

    def _declare_smt_lib_variables(self) -> str:
        variables_declaration = "(declare-fun board_height () Int)\n"
        for i in range(self.n_circuits):
            variables_declaration += "(declare-fun x{} () Int)\n(declare-fun y{} () Int)\n".format(i, i)

        return variables_declaration

    def _declare_smt_lib_constants(self) -> str:
        constants_declaration = "(declare-const n_circuits Int)\n" \
                                "(declare-const board_width Int)\n" \
                                "(declare-const height_lower_bound Int)\n" \
                                "(declare-const height_upper_bound Int)\n"
        for i in range(self.n_circuits):
            constants_declaration += "(declare-const w{} Int)\n(declare-const h{} Int)\n".format(i, i)

        return constants_declaration

    def _get_constants_assertion(self) -> str:
        constants_assertions = "(assert (= n_circuits {}))\n" \
                               "(assert (= board_width {}))\n" \
                               "(assert (= height_lower_bound {}))\n" \
                               "(assert (= height_upper_bound {}))\n".\
            format(self.n_circuits, self.board_width, self.height_lower_bound, self.height_upper_bound)
        for i in range(self.n_circuits):
            constants_assertions += "(assert (= w{} {}))\n(assert (= h{} {}))\n".\
                format(i, self.widths[i], i, self.heights[i])

        return constants_assertions

    def _impose_variables_domain(self) -> str:
        variables_domain_assertions = ""
        for i in range(self.n_circuits):
            variables_domain_assertions += "(assert (>= x{} 0))\n" \
                                           "(assert (>= y{} 0))\n" \
                                           "(assert (<= x{} (- board_width w{})))\n" \
                                           "(assert (<= y{} (- height_upper_bound h{})))\n".format(i, i, i, i, i, i)

        return variables_domain_assertions

    def _impose_effective_board_height(self) -> str:
        effective_board_height_constraint = ""
        for i in range(self.n_circuits):
            effective_board_height_constraint += "(assert (<= y{} (- board_height h{})))\n".format(i, i)

        return effective_board_height_constraint

    @staticmethod
    def _non_overlapping_constraint(i: int, j: int) -> str:
        # x[i] + dx[i] <= x[j] \/ y[i] + dy[i] <= y[j] \/
        # x[j] + dx[j] <= x[i] \/ y[j] + dy[j] <= y[i]
        return "(assert (or (<= (+ x{} w{}) x{}) (<= (+ y{} h{}) y{}) " \
               "(<= (+ x{} w{}) x{}) (<= (+ y{} h{}) y{})))\n".format(i, i, j, i, i, j, j, j, i, j, j, i)

    def _allowable_circuits_positions(self, vertical: bool = False) -> str:
        allowable_circuits_positions_constraints = ""
        variable = "x" if not vertical else "y"
        for i in range(self.n_circuits):
            allowable_circuits_positions_constraints += "(assert (or "
            allowable_circuits_positions_constraints += "(= {}{} 0) ".format(variable, i)
            for j in range(self.n_circuits):
                if i != j:
                    allowable_circuits_positions_constraints += "(= (+ {}{} w{}) {}{}) ".\
                        format(variable, j, j, variable, i)

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

    def to_smt_lib_format(self) -> str:
        smt_lib_instance = self._get_smt_lib_options() + "\n" + self._declare_smt_lib_constants() + "\n" + \
                           self._declare_smt_lib_variables() + "\n" + self._get_constants_assertion() + "\n" + \
                           self._impose_variables_domain()

        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                smt_lib_instance += self._non_overlapping_constraint(i, j)

        # smt_lib_instance += self._allowable_circuits_positions()
        # smt_lib_instance += self._allowable_circuits_positions(vertical=True)
        smt_lib_instance += self.cumulative_constraints_x()
        smt_lib_instance += self.cumulative_constraints_y()
        smt_lib_instance += self._impose_effective_board_height()
        smt_lib_instance += "(assert (= board_height {}))\n".format(self.height_lower_bound)
        smt_lib_instance += "(check-sat)\n(get-model)\n"

        return smt_lib_instance
