"""sumSquare.py Sum Square function
"""
from .problem import Problem

class SumSquare(Problem):
    def evaluate(self, x:list[float]) -> float:
        res = 0.0
        for i in range(self.D):
            res += i * (x[i] ** 2)
        return res