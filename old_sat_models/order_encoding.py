import typing
import itertools
import math

from z3 import *
import numpy as np


def linear_expression_order_encoding(a: typing.List[int], v_indexes: typing.List[int], c: int,
                                     v_lower_bounds: typing.List[int], v_upper_bounds: typing.List[int],
                                     v_order_encoding: typing.List[typing.List[z3.BoolRef]]):
    cumulative_lower_bound = sum([a[i] * v_lower_bounds[v_indexes[i]] if a[i] > 0 else
                                  (a[i]) * v_upper_bounds[v_indexes[i]] for i in range(len(v_indexes))])
    if c < cumulative_lower_bound:
        return [False]

    b_i_combinations = []
    for i in range(len(v_indexes)):
        b_i_lb = a[i] * v_lower_bounds[v_indexes[i]] if a[i] > 0 else (a[i]) * v_upper_bounds[v_indexes[i]]
        b_i_ub = a[i] * v_upper_bounds[v_indexes[i]] if a[i] > 0 else (a[i] * v_lower_bounds[v_indexes[i]])
        b_i_combinations.append(list(range(b_i_lb - 1, b_i_ub + 1)))

    b_cumulative = c - len(v_indexes) + 1
    or_clauses = []

    # extremely inefficient but for linear combinations of two variables with small domains it should be ok
    for b_i_combination in itertools.product(*b_i_combinations):
        or_over_i = []
        if sum(b_i_combination) == b_cumulative:
            for i, b_i in enumerate(b_i_combination):
                c_i = 0
                neg = False
                if a[i] > 0:
                    c_i = math.floor(b_i / a[i])
                else:
                    c_i = math.ceil(b_i / a[i]) - 1
                    neg = True
                if not neg:
                    or_over_i.append(v_order_encoding[v_indexes[i]][c_i - v_lower_bounds[v_indexes[i]] + 1])
                else:
                    or_over_i.append(Not(v_order_encoding[v_indexes[i]][c_i - v_lower_bounds[v_indexes[i]] + 1]))
            or_clauses.append(Or(*or_over_i))

    return or_clauses


def axiom_clauses(v_indexes: typing.List[int], v_lower_bounds: typing.List[int], v_upper_bounds: typing.List[int],
                  v_order_encoding: typing.List[typing.List[z3.BoolRef]]):
    axiom_clauses_ = []
    for v_index in v_indexes:
        axiom_clauses_.append(Not(v_order_encoding[v_index][0]))
        axiom_clauses_.append(v_order_encoding[v_index][v_upper_bounds[v_index] - v_lower_bounds[v_index] + 1])
        for c in range(1, v_upper_bounds[v_index] - v_lower_bounds[v_index] + 2):
            axiom_clauses_.append(Or(Not(v_order_encoding[v_index][c - 1]), v_order_encoding[v_index][c]))

    return axiom_clauses_


def get_order_encoding_variables(v_name: str, v_number: int, v_lower_bounds: typing.List[int],
                                 v_upper_bounds: typing.List[int]):
    return [[Bool(f"p{v_name}_{i + 1}_{_e}") for _e in range(v_lower_bounds[i] - 1, v_upper_bounds[i] + 1)] for i in
            range(v_number)]


def retrieve_v_value(v_order_encoding: typing.List[typing.List[z3.BoolRef]], v_lower_bounds: typing.List[int],
                     v_upper_bounds: typing.List[int],
                     v_index: int, model: z3.ModelRef) -> int:
    model_eval_v_order_encoding = [is_true(model[v_order_encoding[v_index][domain_value]]) for domain_value in
                                   range(v_upper_bounds[v_index] - v_lower_bounds[v_index] + 2)]

    return v_lower_bounds[v_index] - 1 + int(np.argmax(model_eval_v_order_encoding))
