import typing

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
