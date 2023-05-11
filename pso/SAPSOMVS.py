"""SAPSOMVS.py Self-adaptive PSO with multiple velocity strategies
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand
from scipy.stats import cauchy

class SAPSOMVSParticle(CanonicalParticle):
    def __init__(self, D: int) -> None:
        super().__init__(D)
        self.c1:float = None
        self.c2:float = None
        self.w:float = None

class SAPSOMVS:
    def run(self) -> list[tuple[int, int, float]]:
        while self.fecounter < self.maxFEs:
            self._updateSwarm()
            self._updateParameters()
            self._updateGbest()
            self.g += 1
        return self.result

    def __init__(
            self,
            objectFunction:Problem,
            samplePoints:list[float],
            populationSize:int = 20,
            maxGeneration:int = 4000,
            maxFEs:int = 10000,
            initialc1:float = 2.0,
            initialc2:float = 2.0,
            initialw:float = 0.9,
            dtc1:float = 0.3,
            dtc2:float = 0.3,
            dtw:float = 0.2,
            selectionProbability:float = 0.8,
            clipProbability:float = 0.7,
            vmaxPercent:float = 0.2,
            initialSwarm:list[SAPSOMVSParticle] = None
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
        self.initc1 = initialc1
        self.initc2 = initialc2
        self.initw = initialw
        self.dtc1 = dtc1
        self.dtc2 = dtc2
        self.dtw = dtw
        self.sp = selectionProbability
        self.cp = clipProbability
        self.g = 0

        self.lb = objectFunction.lb
        self.ub = objectFunction.ub
        self.vmax = [vmaxPercent * (self.ub[x] - self.lb[x]) for x in range(self.dim)]

        self.swarm = initialSwarm
        if not self.swarm:
            self._initialSwarm()
        
        self.gBestIndex:int = 0
        self._updateGbest()
        self._updateParameters()

    def _initialSwarm(self) -> None:
        self.swarm = []
        for _ in range(self.popSize):
            newParticle = SAPSOMVSParticle(self.dim)
            newParticle.c1 = self.initc1
            newParticle.c2 = self.initc2
            newParticle.w = self.initw
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
        mu1 = 0.1 * (1 - (self.g / self.G) ** 2) + 0.3
        dt1 = 0.1
        mu2 = 0.4 * (1 - (self.g / self.G) ** 2) + 0.2
        dt2 = 0.4
        cauchymu1dt1 = cauchy.rvs(loc=mu1, scale=dt1, size=self.popSize * 2)
        cauchymu2dt2 = cauchy.rvs(loc=mu2, scale=dt2, size=self.popSize * 2)
        gBest = self.swarm[self.gBestIndex]
        for i in range(self.popSize):
            p = self.swarm[i]
            if rand(0,1) > self.sp:
                # uniform random number
                for d in range(self.dim):
                    p.v[d] = p.w * p.v[d] + p.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                                + p.c2 * rand(0,1) * (gBest.pbest[d] - p.x[d])
            else:
                # cauchy random number
                cauchy1 = cauchymu1dt1[i]
                cauchy2 = cauchymu1dt1[i + 1]
                if rand(0,1) > 0.5:
                    cauchy1 = cauchymu2dt2[i]
                    cauchy2 = cauchymu2dt2[i + 1]
                for d in range(self.dim):
                    p.v[d] = p.w * p.v[d] + p.c1 * cauchy1 * (p.pbest[d] - p.x[d]) \
                                + p.c2 * cauchy2 * (gBest.pbest[d] - p.x[d])
            for d in range(self.dim):
                p.v[d] = max(-self.vmax[d], min(self.vmax[d], p.v[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                if p.x[d] > self.ub[d] or p.x[d] < self.lb[d]:
                    if rand(0,1) > self.cp:
                        p.x[d] = rand(self.lb[d], self.ub[d])
                    else:
                        p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            # evaluate fitness and update pbest
            p.fx = self.f(p.x)
            if self.fitter(p.fx, p.fpbest):
                p.updatePbest()

    def _updateParameters(self) -> None:
        # calculate weight
        fvals = [p.fx for p in self.swarm]
        fmax = max(fvals)
        weights = [abs(f - fmax) for f in fvals]
        total = sum(weights)
        weights = [ (w / total) for w in weights]
        # average values
        w = sum([weights[i] * self.swarm[i].w for i in range(self.popSize)])
        c1 = sum([weights[i] * self.swarm[i].c1 for i in range(self.popSize)])
        c2 = sum([weights[i] * self.swarm[i].c2 for i in range(self.popSize)])
        cauchyw = cauchy.rvs(loc = w, scale=self.dtw, size=self.popSize)
        cauchyc1 = cauchy.rvs(loc=c1, scale=self.dtc1, size=self.popSize)
        cauchyc2 = cauchy.rvs(loc=c2, scale=self.dtc2, size=self.popSize)
        for i in range(self.popSize):
            p = self.swarm[i]
            p.w = cauchyw[i]
            p.c1 = cauchyc1[i]
            p.c2 = cauchyc2[i]
            if p.w > 1.0:
                p.w = rand(0,1)
            elif p.w < 0:
                p.w = rand(0,0.1)
            if p.c1 > 4.0:
                p.c1 = rand(0,4)
            elif p.c1 < 0:
                p.c1 = rand(0,1)
            if p.c2 > 4.0:
                p.c2 = rand(0,4)
            elif p.c2 < 0:
                p.c2 = rand(0,1)

    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        if self.fecounter in self.samplePoints:
            self.result.append((self.fecounter, self.g, self.err(self.swarm[self.gBestIndex].fpbest)))
        return self.evaluate(x)