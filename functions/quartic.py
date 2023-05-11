"""quartic.py Quartic function with noise
"""
from .problem import Problem
from random import random as rand

class QuarticWithNoise(Problem):
    def evaluate(self, solution:list[float]) -> float:
        res = 0.0
        for i in range(self.D):
            res += i * (solution[i] ** 4)
        return res + rand()