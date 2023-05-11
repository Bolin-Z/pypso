""" EPSO.py Extraordinariness PSO
"""
from .common import BaseParticle
from functions.problem import Problem
from random import uniform as rand
from typing import Callable

class ExtraParticle(BaseParticle):
    def __init__(self, D:int, fitter:Callable[[float,  float], bool]) -> None:
        super().__init__(D)
        self.fitter = fitter
    def __lt__(self, other:"ExtraParticle") -> bool:
        return self.fitter(other.fpbest, self.fpbest)
    def __eq__(self, other:"ExtraParticle") -> bool:
        return self.fpbest == other.fpbest

class EPSO:
    def run(self) -> list[tuple[int, int, float]]:
        while self.fecounter < self.maxFEs:
            self.swarm.sort(reverse=True)
            self._updateSwarm()
            self.g += 1
        return self.result

    def __init__(
            self,
            objectFunction:Problem,
            samplePoints:list[float],
            populationSize:int = 20,
            maxGeneration:int = 4000,
            maxFEs:int = 10000,
            c:float = 0.3,
            alpha:float = 0.8,
            initialSwarm:list[ExtraParticle] = None
        ) -> None:
        
        self.result:list[tuple[int, int, float]] = []

        self.fecounter:int = 0
        self.maxFEs = maxFEs
        self.samplePoints = [self.maxFEs * p for p in samplePoints]
        self.evaluate = objectFunction.evaluate
        self.fitter = objectFunction.fitter
        self.err = objectFunction.err

        self.dim = objectFunction.D
        self.popSize = populationSize
        self.G = maxGeneration
        self.c = c
        self.alpha = alpha
        self.Tup = round(self.alpha * self.popSize)
        self.g = 0

        self.lb = objectFunction.lb
        self.ub = objectFunction.ub

        self.swarm = initialSwarm
        if not self.swarm:
            self._initialSwarm()

    def _initialSwarm(self) -> None:
        self.swarm = []
        for _ in range(self.popSize):
            newParticle = ExtraParticle(self.dim, self.fitter)
            for d in range(self.dim):
                newParticle.x[d] = rand(self.lb[d], self.ub[d])
            newParticle.fx = self.f(newParticle.x)
            newParticle.updatePbest()
            self.swarm.append(newParticle)

    def _updateSwarm(self) -> None:
        for i in range(self.popSize):
            p = self.swarm[i]
            exemplarIdx = round(rand(0,1) * self.popSize)
            if exemplarIdx < self.Tup:
                # learn from exemplar
                exemplar = self.swarm[exemplarIdx]
                for d in range(self.dim):
                    p.x[d] = p.x[d] + self.c * (exemplar.pbest[d] - p.x[d])
                    p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            else:
                # random search
                for d in range(self.dim):
                    p.x[d] = rand(self.lb[d], self.ub[d])
            # evaluate fitness and update pbest
            p.fx = self.f(p.x)
            if (self.fitter(p.fx, p.fpbest)):
                p.updatePbest()

    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        if self.fecounter in self.samplePoints:
            self.result.append((self.fecounter, self.g, self.err(self.swarm[0].fpbest)))
        return self.evaluate(x)