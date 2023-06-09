"""canonicalPSO.py
    Canonical PSO
    Original PSO
"""
from .common import BaseParticle
from functions.problem import Problem
from random import uniform as rand

class CanonicalParticle(BaseParticle):
    def __init__(self, D:int) -> None:
        super().__init__(D)
        self.v = [0.0 for _ in range(D)]

class CanonicalPSO:
    def run(self) -> tuple[float, list[float]]:
        while self.g < self.G:
            self._updateSwarm()
            self._updateGbest()
            self.g += 1
        gbest = self.swarm[self.gBestIndex]
        return (gbest.fpbest, gbest.pbest)
    
    def __init__(
            self,
            objectFunction:Problem,
            populationSize:int = 20, 
            maxGeneration:int = 4000,
            c1:float = 2.0,
            c2:float = 2.0,
            w:float = 0.9,
            vmaxPercent:float = 0.2,
            initialSwarm:list[CanonicalParticle] = None
        ) -> None:

        self.f = objectFunction.evaluate
        self.fitter = objectFunction.fitter
        
        self.dim = objectFunction.D
        self.popSize = populationSize
        self.G = maxGeneration
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.g = 0

        self.lb = objectFunction.lb
        self.ub = objectFunction.ub        
        self.vmax = [vmaxPercent * (self.ub[x] - self.lb[x]) for x in range(self.dim)]
        
        self.swarm = initialSwarm
        if not self.swarm:
            self._initialSwarm()
        
        self.gBestIndex:int = 0
        self._updateGbest()

    def _initialSwarm(self) -> None:
        self.swarm = []
        for _ in range(self.popSize):
            newParticle = CanonicalParticle(self.dim)
            for d in range(self.dim):
                newParticle.x[d] = rand(self.lb[d], self.ub[d])
                newParticle.v[d] = rand(-self.vmax[d], self.vmax[d])
            newParticle.fx = self.f(newParticle.x)
            newParticle.updatePbest()
            self.swarm.append(newParticle)
    
    def _updateGbest(self) -> None:
        for i in range(self.popSize):
            if self.fitter(self.swarm[i].fpbest, self.swarm[self.gBestIndex].fpbest):
                self.gBestIndex = i
    
    def _updateSwarm(self) -> None:
        gBest = self.swarm[self.gBestIndex]
        for i in range(self.popSize):
            p = self.swarm[i]
            for d in range(self.dim):
                # update velocity
                p.v[d] = self.w * p.v[d] + self.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                            + self.c2 * rand(0,1) * (gBest.pbest[d] - p.x[d])
                p.v[d] = max(-self.vmax[d], min(self.vmax[d], p.v[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            # evaluate fitness and update pbest
            p.fx = self.f(p.x)
            if(self.fitter(p.fx, p.fpbest)):
                p.updatePbest()

class OriginalPSO(CanonicalPSO):
    def __init__(
            self, 
            objectFunction: Problem, 
            populationSize: int = 20, 
            maxGeneration: int = 4000, 
            c1: float = 2.0, 
            c2: float = 2.0, 
            vmaxPercent: float = 0.2, 
            initialSwarm: list[CanonicalParticle] = None
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