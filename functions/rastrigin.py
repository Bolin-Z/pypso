"""rastrigin.py Rastrigin Function
"""
from .problem import Problem
from math import cos, pi

class Rastrigin(Problem):
    def evaluate(self, solution: list[float]) -> float:
        res = 0.0
        for x in solution:
            res += ((x ** 2) - 10 * cos(2 * pi * x))
        return 10 * self.D + res