"""FDRPSO.py Fitness-distance-ratio based PSO
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand

class FDRPSO:
    def run(self) -> tuple[float, list[float]]:
        while self.g < self.G:
            self._updateSwarm()
            self._updateGbest()
            self._updateInertiaWeight()
            self.g += 1
        gbest = self.swarm[self.gBestIndex]
        return (gbest.fpbest, gbest.pbest)

    def __init__(
            self,
            objectFunction:Problem,
            populationSize:int = 20,
            maxGeneration:int = 4000,
            c1:float = 1.0,
            c2:float = 1.0,
            c3:float = 2.0,
            wmin:float = 0.4,
            wmax:float = 0.9,
            vmaxPercent:float = 0.2,
            initialSwarm:list[CanonicalParticle] = None
        ) -> None:
        
        self.fecounter:int = 0
        self.evaluate = objectFunction.evaluate
        self.fitter = objectFunction.fitter
        self.theta = -1 if objectFunction.minimize else 1

        self.dim = objectFunction.D
        self.popSize = populationSize
        self.G = maxGeneration
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
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
                # find nbest in dimension d
                nbestIdx = self._FDR(i, d)
                nbest = self.swarm[nbestIdx]
                # update velocity
                p.v[d] = self.w * p.v[d] + self.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                            + self.c2 * rand(0,1) * (gBest.pbest[d] - p.x[d]) \
                                + self.c3 * rand(0,1) * (nbest.pbest[d] - p.x[d])
                p.v[d] = max(-self.vmax[d], min(p.v[d], self.vmax[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                p.x[d] = max(self.lb[d], min(p.x[d], self.ub[d]))
            # evaluate fitness and update pbest
            p.fx = self.f(p.x)
            if self.fitter(p.fx, p.fpbest):
                p.updatePbest()

    def _updateInertiaWeight(self) -> None:
        self.w = self.wmax - (self.wmax - self.wmin) * (self.g / self.G)
    
    def _FDR(self, targetIdx:int, d:int) -> int:
        # find nbest index for particle targetIdx of dimension d
        p = self.swarm[targetIdx]
        fdrnum = self.theta * (p.fpbest - p.fx)
        fdrden = abs(p.pbest[d] - p.x[d])
        nbestIdx = targetIdx
        for j in range(self.popSize):
            n = self.swarm[j]
            newfdrnum = self.theta * (n.fpbest - p.fx)
            newfdrden = abs(n.pbest[d] - p.x[d])
            if newfdrnum * fdrden > fdrnum * newfdrden:
                fdrnum = newfdrnum
                fdrden = newfdrden
                nbestIdx = j
        return nbestIdx

    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        return self.evaluate(x)