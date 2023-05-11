"""griewank.py Griewand function
"""
from .problem import Problem
from math import sqrt, cos

class Griewand(Problem):
    def evaluate(self, x:list[float]) -> float:
        totalSum = 0.0
        totalProduct = 1.0
        for i in range(self.D):
            totalSum += ((x[i] ** 2) / 4000)
            totalProduct *= cos(x[i] / sqrt(i))
        return totalSum - totalProduct + 1
