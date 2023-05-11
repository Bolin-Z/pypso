"""easom.py Easom function
"""
from .problem import Problem
from math import cos, exp, pi

class Easom(Problem):
    def evaluate(self, solution:list[float]) -> float:
        x1 = solution[0]
        x2 = solution[1]
        return -cos(x1) * cos(x2) * exp(-((x1 - pi) ** 2) - ((x2 - pi) ** 2))