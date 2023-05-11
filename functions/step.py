"""step.py Step function
"""
from .problem import Problem
from math import floor

class Step(Problem):
    def evaluate(self, solution:list[float]) -> float:
        res = 0.0
        for x in solution:
            res += (floor(x + 0.5)) ** 2
        return res
    def err(self, fval: float) -> float:
        return abs(fval - 0.0)