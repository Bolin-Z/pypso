"""sphere.py Sphere function
"""

from .problem import Problem
from typing import List

class Sphere(Problem):
    def evaluate(self, solution: list[float]) -> float:
        res = 0
        for i in range(self.D):
            res += solution[i] ** 2
        return res