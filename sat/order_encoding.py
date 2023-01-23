import typing
import math

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

    def __str__(self) -> str:
        return self.name + "[" + str(self.lower_bound) + ", " + str(self.upper_bound) + "]"

    def actual_value(self, model: z3.ModelRef) -> int:
        model_eval_order_encoding_booleans = [is_true(model[self.order_encoding_booleans[domain_val]]) for domain_val in
                                              range(self.upper_bound - self.lower_bound + 2)]

        return self.lower_bound - 1 + np.argmax(model_eval_order_encoding_booleans)

    def get_axiom_clauses(self) -> typing.List[typing.List[BoolRef]]:
        axiom_clauses_ = list()
        axiom_clauses_.append(Not(self.order_encoding_booleans[0]))
        axiom_clauses_.append(self.order_encoding_booleans[self.upper_bound - self.lower_bound + 1])
        for c in range(1, self.upper_bound - self.lower_bound + 2):
            axiom_clauses_.append(Or(Not(self.order_encoding_booleans[c - 1]), self.order_encoding_booleans[c]))

        return axiom_clauses_

    @staticmethod
    def linear_inequality_constraint(a: int, var_a: "OrderEncodedVariable", b: int, var_b: "OrderEncodedVariable", c: int) \
            -> typing.List[BoolRef]:
        constraints = list()
        cumulative_lower_bound = a * var_a.lower_bound if a > 0 else a * var_a.upper_bound
        cumulative_lower_bound += b * var_b.lower_bound if b > 0 else b * var_b.upper_bound

        if c < cumulative_lower_bound:
            return [False]

        a_range = range(a * var_a.lower_bound - 1, a * var_a.upper_bound + 1) if a > 0 else \
            range(a * var_a.upper_bound - 1, a * var_a.lower_bound + 1)
        b_range = range(b * var_b.lower_bound - 1, b * var_b.upper_bound + 1) if b > 0 else \
            range(b * var_b.upper_bound - 1, b * var_b.lower_bound + 1)

        for a_val in a_range:
            for b_val in b_range:
                if a_val + b_val == c - 1:
                    or_clauses = []
                    c_a = math.floor((a_val / a)) if a > 0 else math.ceil(a_val / a) - 1
                    c_b = math.floor((b_val / b)) if b > 0 else math.ceil(b_val / b) - 1

                    or_clauses.append(Not(var_a.order_encoding_booleans[c_a - var_a.lower_bound + 1]) if a <= 0 else
                                      var_a.order_encoding_booleans[c_a - var_a.lower_bound + 1])

                    or_clauses.append(Not(var_b.order_encoding_booleans[c_b - var_b.lower_bound + 1]) if b <= 0 else
                                      var_b.order_encoding_booleans[c_b - var_b.lower_bound + 1])

                    constraints.append(Or(*or_clauses))

        return constraints
