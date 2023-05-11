"""AIWPSO.py Adaptive inertia weight PSO
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand
from random import gauss as gauss
from random import randrange as randrange

class AIWPSO:
    def run(self) -> list[tuple[int, int, float]]:
        while self.fecounter < self.maxFEs:
            # count number of improved particles in this generation
            self.successCount = 0
            self._updateSwarm()
            self._updateGbestandGworst()
            self._updateInertiaWeight()
            self._mutatedAndReplace()
            self.g += 1
        return self.result

    def __init__(
            self,
            objectFunction:Problem,
            populationSize:int = 20,
            maxGeneration:int = 4000,
            maxFEs:int = 10000,
            c1:float = 1.49445,
            c2:float = 1.49445,
            wmin:float = 0.0,
            wmax:float = 1.0,
            vmaxPercent:float = 0.2,
            initialSwarm:list[CanonicalParticle] = None
        ) -> None:

        self.result:list[tuple[int, int, float]] = []

        self.fecounter:int = 0
        self.maxFEs = maxFEs
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
        self.gWorstIndex:int = 0
        self._updateGbestandGworst()

        self.successCount:int = 0
            
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

    def _updateGbestandGworst(self) -> None:
        for i in range(self.popSize):
            if self.fitter(self.swarm[i].fpbest, self.swarm[self.gBestIndex].fpbest):
                self.gBestIndex = i
            elif self.fitter(self.swarm[self.gWorstIndex].fpbest, self.swarm[i].fpbest):
                self.gWorstIndex = i

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
                # increase the number of improved particles
                self.successCount += 1
                p.updatePbest()

    def _updateInertiaWeight(self) -> None:
        ps = self.successCount / self.popSize
        self.w = self.wmin + (self.wmax - self.wmin) * ps

    def _mutatedAndReplace(self) -> None:
        gBest = self.swarm[self.gBestIndex]
        gWorst = self.swarm[self.gWorstIndex]

        mutatedim = randrange(0, self.dim)
        sigma = (1 - self.g / self.G) * (self.ub[mutatedim] - self.lb[mutatedim])
        # copy
        gWorst.pbest = [x for x in gBest.pbest]
        gWorst.pbest[mutatedim] = gWorst.pbest[mutatedim] + gauss(0, sigma)
        if gWorst.pbest[mutatedim] > self.ub[mutatedim]:
            gWorst.pbest[mutatedim] = self.ub[mutatedim]
        elif gWorst.pbest[mutatedim] < self.lb[mutatedim]:
            gWorst.pbest[mutatedim] = self.lb[mutatedim]
        
        gWorst.fpbest = self.f(gWorst.pbest)
        if self.fitter(gWorst.fpbest, gBest.fpbest):
            self.gBestIndex = self.gWorstIndex
    
    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        t = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in t:
            if self.fecounter == self.maxFEs * i:
                self.result.append((self.fecounter, self.g, self.err(self.swarm[self.gBestIndex].fpbest)))
        return self.evaluate(x)