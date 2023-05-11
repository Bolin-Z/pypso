"""rosenbrock Rosenbrock function
"""
from .problem import Problem

class Rosenbrock(Problem):
    def evaluate(self, x:list[float]) -> float:
        res = 0.0
        for i in range(self.D - 1):
            res += (100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2)
        return res
    def err(self, fval: float) -> float:
        return abs(fval - 0.0)