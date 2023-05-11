"""ackley.py Ackley function
"""
from .problem import Problem
from math import pi, exp, sqrt, cos

class Ackley(Problem):
    def evaluate(self, solution:list[float]) -> float:
        a = 20
        b = 0.2
        c = 2 * pi
        d = self.D
        alpha = 0.0
        beta = 0.0
        for x in solution:
            alpha += (x ** 2)
            beta += cos(c * x)
        return - a * exp(-b * sqrt(alpha / d)) - exp(beta / d) + a + exp(1)