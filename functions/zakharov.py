"""zakharov.py Zakharov function
"""
from .problem import Problem

class Zakharov(Problem):
    def evaluate(self, x:list[float]) -> float:
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        for i in range(self.D):
            sum1 += x[i] ** 2
            sum2 += 0.5 * i * x[i]
            sum3 += 0.5 * i * x[i]
        return sum1 + (sum2 ** 2) + (sum3 ** 4)
    def err(self, fval: float) -> float:
        return abs(fval - 0.0)