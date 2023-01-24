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


class SMTModel:
    def __init__(self, n_circuits: int, board_width: int, widths: typing.List[int], heights: typing.List[int],
                 height_lower_bound: int, board_height: int, time_limit_ms: int, allow_rotation: bool = False):
        self.n_circuits = n_circuits
        self.board_width = board_width
        self.board_height = board_height
        self.widths = widths
        self.heights = heights
        self.height_lower_bound = height_lower_bound
        self.time_limit_ms = time_limit_ms
        self.allow_rotation = allow_rotation
        print("Time limit set to: {}".format(self.time_limit_ms))

    @staticmethod
    def from_instance_json(json_filepath: str, board_height: int, allow_rotation: bool, time_limit_ms: int) \
            -> "SMTModel":
        with open(json_filepath, "r") as f:
            instance = json.load(f)
        instance.pop("height_upper_bound")

        return SMTModel(**instance, board_height=board_height, time_limit_ms=time_limit_ms,
                        allow_rotation=allow_rotation)

    def _get_smt_lib_options(self) -> str:
        return "(set-option :produce-models true)\n(set-option :timeout {})\n(set-logic QF_LIA)\n". \
            format(self.time_limit_ms)

    def _declare_smt_lib_variables(self) -> str:
        variables_declaration = ""
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
                variables_domain_assertions += "(assert (<= y{} (- {} rh{})))\n".format(i, self.board_height, i)
            else:
                variables_domain_assertions += "(assert (<= x{} {}))\n".format(i, self.board_width - self.widths[i])
                variables_domain_assertions += "(assert (<= y{} {}))\n".format(i, self.board_height - self.heights[i])

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
            constraints += "(assert (<= {} {}))\n".format(sum_expr, self.board_height)

        return constraints

    def cumulative_constraints_y(self):
        constraints = ""
        early = 0
        late = self.board_height + max(self.heights)
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

    def _large_circuits_pair(self):
        constraints = ""
        for i in range(self.n_circuits):
            for j in range(i + 1, self.n_circuits):
                if self.widths[i] + self.widths[j] > self.board_width:
                    if self.allow_rotation:
                        # not rot(i) not rot(j), focus on width
                        constraints += "(assert (=> (and (not r{i}) (not r{j})) " \
                                       "(or (<= (+ y{i} {height_i}) y{j}) (>= y{i} (+ y{j} {height_j})))))\n". \
                            format(i=i, j=j, height_i=self.heights[i], height_j=self.heights[j])
                    else:
                        constraints += "(assert (or (<= (+ y{i} {height_i}) y{j}) (>= y{i} (+ y{j} {height_j}))))\n". \
                            format(i=i, j=j, height_i=self.heights[i], height_j=self.heights[j])
                if self.heights[i] + self.heights[j] > self.board_height:
                    if self.allow_rotation:
                        # not rot(i) not rot(j), focus on height
                        constraints += "(assert (=> (and (not r{i}) (not r{j})) " \
                                       "(or (<= (+ x{i} {width_i}) x{j}) (>= x{i} (+ x{j} {width_j})))))\n". \
                            format(i=i, j=j, width_i=self.widths[i], width_j=self.widths[j])
                    else:
                        constraints += "(assert (or (<= (+ x{i} {width_i}) x{j}) (>= x{i} (+ x{j} {width_j}))))\n". \
                            format(i=i, j=j, width_i=self.widths[i], width_j=self.widths[j])
                if self.allow_rotation and self.widths[i] + self.heights[j] > self.board_width:
                    # rot(i) not rot(j), focus on width
                    constraints += "(assert (=> (and r{i} (not r{j})) " \
                                   "(or (<= (+ y{i} {width_i}) y{j}) (>= y{i} (+ y{j} {height_j})))))\n". \
                        format(i=i, j=j, width_i=self.widths[i], height_j=self.heights[j])
                if self.allow_rotation and self.heights[i] + self.widths[j] > self.board_width:
                    # not rot(i) rot(j), focus on width
                    constraints += "(assert (=> (and (not r{i}) r{j}) " \
                                   "(or (<= (+ y{i} {height_i}) y{j}) (>= y{i} (+ y{j} {width_j})))))\n". \
                        format(i=i, j=j, height_i=self.heights[i], width_j=self.widths[j])
                if self.allow_rotation and self.widths[i] + self.heights[j] > self.board_height:
                    # rot(i) not rot(j), focus on height
                    constraints += "(assert (=> (and r{i} (not r{j})) " \
                                   "(or (<= (+ x{i} {height_i}) x{j}) (>= x{i} (+ x{j} {width_j})))))\n". \
                        format(i=i, j=j, height_i=self.heights[i], width_j=self.widths[j])
                if self.allow_rotation and self.heights[i] + self.widths[j] > self.board_height:
                    # not rot(i) rot(j), focus on height
                    constraints += "(assert (=> (and (not r{i}) r{j}) " \
                                   "(or (<= (+ x{i} {width_i}) x{j}) (>= x{i} (+ x{j} {height_j})))))\n". \
                        format(i=i, j=j, width_i=self.widths[i], height_j=self.heights[j])
                if self.allow_rotation and self.widths[i] + self.widths[j] > self.board_height:
                    # rot(i) rot(j), focus on width
                    constraints += "(assert (=> (and r{i} r{j}) " \
                                   "(or (<= (+ x{i} {height_i}) x{j}) (>= x{i} (+ x{j} {height_j})))))\n". \
                        format(i=i, j=j, height_i=self.heights[i], height_j=self.heights[j])
                if self.allow_rotation and self.heights[i] + self.heights[j] > self.board_width:
                    # rot(i) rot(j), focus on height
                    constraints += "(assert (=> (and r{i} r{j}) " \
                                   "(or (<= (+ y{i} {width_i}) y{j}) (>= y{i} (+ y{j} {width_j})))))\n". \
                        format(i=i, j=j, width_i=self.widths[i], width_j=self.widths[j])

        return constraints

    def lexicographic_ordering_symmetry_breaking(self):
        constraints = ""
        for i in range(self.n_circuits):
            if not self.allow_rotation:
                reflected_x_positions = ["(- {} {} {})".format(self.board_width, self.widths[i], "x{}".format(i)) for i
                                         in
                                         range(self.n_circuits)]
                reflected_y_positions = ["(- {} {} {})".format(self.board_height, self.heights[i], "y{}".format(i))
                                         for i in range(self.n_circuits)]
                constraints += lexicographic_ordering_ror("aux_lex_x_{}_".format(i),
                                                          ["x{}".format(i) for i in range(self.n_circuits)],
                                                          reflected_x_positions)
                constraints += lexicographic_ordering_ror("aux_lex_y_{}_".format(i),
                                                          ["y{}".format(i) for i in range(self.n_circuits)],
                                                          reflected_y_positions)
            else:
                reflected_x_positions = ["(- {} rw{} {})".format(self.board_width, i, "x{}".format(i)) for i
                                         in range(self.n_circuits)]
                reflected_y_positions = ["(- {} rh{} {})".format(self.board_height, i, "y{}".format(i))
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

        if self.allow_rotation:
            smt_lib_instance += self._enforce_rotation() + "\n"

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

        smt_lib_instance += "(check-sat)\n"

        return smt_lib_instance
