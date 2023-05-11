"""problem.py problem template for optimization problems
"""
from abc import ABC, abstractmethod
from typing import List

class Problem(ABC):
    def __init__(self, lb:List[float], ub:List[float], minimize:bool = True) -> None:
        self.lb = lb
        self.ub = ub
        self.D = len(lb)
        self.minimize = minimize
    
    def fitter(self, fvalA:float, fvalB:float) -> bool:
        """
        return whether fvalA is better than fvalB
        """
        if self.minimize:
            return fvalA < fvalB
        else:
            return fvalA > fvalB

    @abstractmethod
    def evaluate(self, solution:list[float]) -> float:
        """
        object function
        """
        pass

    @abstractmethod
    def err(self, fval:float) -> float:
        """
        abs(fval - f*)
        """
        pass