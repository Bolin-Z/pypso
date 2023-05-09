"""CLPSO.py Comprehensive learning PSO
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand, randrange
from math import exp

class CLParticle(CanonicalParticle):
    def __init__(self, D:int, pc:float) -> None:
        super().__init__(D)
        self.pc:float = pc
        self.fi:list[int] = [0 for _ in range(D)]
        self.stagnate = 0

class CLPSO:
    def run(self) -> tuple[float, list[float]]:
        while self.g < self.G:
            self._updateInertiaWeight()
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
            c:float = 2.0,
            wmin:float = 0.4,
            wmax:float = 0.9,
            stagnateMax:int = 7,
            vmaxPercent:float = 0.2,
            initialSwarm:list[CLParticle] = None
        ) -> None:
        
        self.fecounter:int = 0
        self.evaluate = objectFunction.evaluate
        self.fitter = objectFunction.fitter

        self.dim = objectFunction.D
        self.popSize = populationSize
        self.G = maxGeneration
        self.c = c
        self.wmin = wmin
        self.wmax = wmax
        self.stMax = stagnateMax
        self.w = self.wmax
        self.g = 0

        self.lb = objectFunction.lb
        self.ub = objectFunction.ub
        self.vmax = [vmaxPercent * (self.ub[x] - self.lb[x]) for x in range(self.dim)]

        self.swarm = initialSwarm
        if not self.swarm:
            self._initialSwarm()
        
        self.gBestIndex:int = 0
        self._updateGbest()

        for i in range(self.popSize):
            self._constructFi(i)
    
    def _initialSwarm(self) -> None:
        self.swarm = []
        for i in range(self.popSize):
            pc = 0.05 + 0.45 * (exp(10 * i / (self.popSize - 1)) - 1) / (exp(10) - 1)
            newParticle = CLParticle(self.dim, pc)
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
        for i in range(self.popSize):
            p = self.swarm[i]
            # check stagnate
            if p.stagnate >= self.stMax:
                self._constructFi(i)
                p.stagnate = 0
            inRange = True
            for d in range(self.dim):
                # update velocity
                exemplar = self.swarm[p.fi[d]]
                p.v[d] = self.w * p.v[d] + self.c * rand(0,1) * (exemplar.pbest[d] - p.x[d])
                p.v[d] = max(-self.vmax[d], min(p.v[d], self.vmax[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                if p.x[d] < self.lb[d] or p.x[d] > self.ub[d]:
                    inRange = False
            if inRange:
                # evaluate fitness and update pbest
                p.fx = self.f(p.x)
                p.stagnate += 1
                if self.fitter(p.fx, p.fpbest):
                    p.updatePbest()
                    p.stagnate = 0

    def _constructFi(self, idx:int) -> None:
        p = self.swarm[idx]
        allPbest = True
        for d in range(self.dim):
            p.fi[d] = idx
            if rand(0,1) < p.pc:
                # choose two other particles
                idx1 = randrange(0, self.popSize)
                while idx1 == idx:
                    idx1 = randrange(0, self.popSize)
                idx2 = randrange(0, self.popSize)
                while idx2 == idx or idx2 == idx1:
                    idx2 = randrange(0, self.popSize)
                if self.fitter(self.swarm[idx1].fpbest, self.swarm[idx2].fpbest):
                    p.fi[d] = idx1
                else:
                    p.fi[d] = idx2
                allPbest = False
        if allPbest:
            mutate = randrange(0, self.dim)
            idx3 = randrange(0, self.popSize)
            while idx3 == idx:
                idx3 = randrange(0, self.popSize)
            p.fi[mutate] = idx3
            
    def _updateInertiaWeight(self) -> None:
        self.W = self.wmax - (self.wmax - self.wmin) * (self.g / self.G)

    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        return self.evaluate(x)