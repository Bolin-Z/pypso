"""sphere.py Sphere function
"""
from .problem import Problem

class Sphere(Problem):
    def evaluate(self, solution: list[float]) -> float:
        res = 0
        for i in range(self.D):
            res += solution[i] ** 2
        return res
    def err(self, fval:float) -> float:
        return abs(fval - 0.0)