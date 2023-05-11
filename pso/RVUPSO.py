"""RVUPSO.py Relaxation velocity update PSO
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand

class RVUParticle(CanonicalParticle):
    def __init__(self, D: int) -> None:
        super().__init__(D)
        self.updateV:bool = False

class RVUPSO:
    def run(self) -> list[tuple[int, int, float]]:
        while self.fecounter < self.maxFEs:
            self._updateSwarm()
            self._updateGbest()
            self._updateInertiaWeight()
            self.g += 1
        return self.result

    def __init__(
            self,
            objectFunction:Problem,
            samplePoints:list[float],
            populationSize:int = 30,
            maxGeneration:int = 4000,
            maxFEs:int = 10000,
            c1:float = 2.0,
            c2:float = 2.0,
            wmin:float = 0.4,
            wmax:float = 0.9,
            vmaxPercent:float = 1.0,
            initialSwarm:list[RVUParticle] = None
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
        self.c1 = c1
        self.c2 = c2
        self.wmin = wmin
        self.wmax = wmax
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

    def _initialSwarm(self) -> None:
        self.swarm = []
        for _ in range(self.popSize):
            newParticle = RVUParticle(self.dim)
            for d in range(self.dim):
                newParticle.x[d] = rand(self.lb[d], self.ub[d])
                newParticle.v[d] = rand(-self.vmax[d], self.vmax[d])
            newParticle.fx = self.f(newParticle.x)
            newParticle.updateV = False
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
                if p.updateV:
                    # update velocity
                    p.v[d] = self.w * p.v[d] + self.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                                + self.c2 * rand(0,1) * (gBest.pbest[d] - p.x[d])
                    p.v[d] = max(-self.vmax[d], min(self.vmax[d], p.v[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            # evaluate fitness and update pbest
            oldFx = p.fx
            p.fx = self.f(p.x)
            if self.fitter(oldFx, p.fx):
                p.updateV = True
            else:
                p.updateV = False
            if self.fitter(p.fx, p.fpbest):
                p.updatePbest()

    def _updateInertiaWeight(self) -> None:
        self.w = self.wmax - (self.wmax - self.wmin) * (self.g / self.G)

    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        if self.fecounter in self.samplePoints:
            self.result.append((self.fecounter, self.g, self.err(self.swarm[self.gBestIndex].fpbest)))
        return self.evaluate(x)