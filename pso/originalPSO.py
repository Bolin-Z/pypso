"""originalPSO.py 
"""
from .particles import BasicParticle
from functions.problem import Problem
from .canonicalPSO import CanonicalPSO

class OriginalPSO(CanonicalPSO):
    def __init__(
            self, 
            objectFunction: Problem, 
            populationSize: int = 20, 
            maxGeneration: int = 1000, 
            c1: float = 2, 
            c2: float = 2, 
            vmaxPercent: float = 0.2, 
            initialSwarm: list[BasicParticle] = None
        ) -> None:
        super().__init__(
            objectFunction, 
            populationSize, 
            maxGeneration, 
            c1, 
            c2, 
            1.0, 
            vmaxPercent, 
            initialSwarm
        )