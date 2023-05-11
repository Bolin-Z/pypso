"""schaffer.py Schaffer's f6 function 2-D
"""

from .problem import Problem
from math import sqrt, sin

class Schaffer(Problem):
    def evaluate(self, solution: list[float]) -> float:
        x = solution[0]
        y = solution[1]
        return 0.5 + ((sin(sqrt(x**2 - y**2)))**2 - 0.5) / ((1 + 0.001*(x**2 + y**2)) ** 2)
    def err(self, fval: float) -> float:
        return abs(fval - 0.0)