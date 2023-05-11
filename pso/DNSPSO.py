"""DNSPSO.py Diversity enhancing mechanism and neighborhood search strategies PSO
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand, sample

class DNSPSO:
    def run(self) -> list[tuple[int, int, float]]:
        while self.fecounter < self.maxFEs:
            self._updateSwarm()
            self._updateGbest()
            self._neighborhoodSearch()
            self.g += 1
        return self.result

    def __init__(
            self,
            objectFunction:Problem,
            samplePoints:list[float],
            populationSize:int = 20,
            maxGeneration:int = 4000,
            maxFEs:int = 10000,
            c1:float = 1.49618,
            c2:float = 1.49618,
            w:float = 0.7298,
            k:int = 2,
            pr:float = 0.9,
            pns:float = 0.6,
            vmaxPercent:float = 0.2,
            initialSwarm:list[CanonicalParticle] = None
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
        self.w = w
        self.k = k
        self.pr = pr
        self.pns =pns
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

    def _updateSwarm(self) -> None:
        gBest = self.swarm[self.gBestIndex]
        for i in range(self.popSize):
            p = self.swarm[i]
            tp = [t for t in p.x]
            for d in range(self.dim):
                # update velocity
                p.v[d] = self.w * p.v[d] + self.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                            + self.c2 * rand(0,1) * (gBest.pbest[d] - p.x[d])
                p.v[d] = max(-self.vmax[d], min(self.vmax[d], p.x[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            # evaluate fitness
            p.fx = self.f(p.x)
            # generate trial particle
            for d in range(self.dim):
                if rand(0,1) < self.pr:
                    tp[d] = p.x[d]
            # test the trial particle
            ftp = self.f(tp)
            if self.fitter(ftp, p.fx):
                p.x = [i for i in tp]
                p.fx = ftp
            if self.fitter(p.fx, p.fpbest):
                p.updatePbest()

    def _updateGbest(self) -> None:
        for i in range(self.popSize):
            if self.fitter(self.swarm[i].fpbest, self.swarm[self.gBestIndex].fpbest):
                self.gBestIndex = i

    def _neighborhoodSearch(self) -> None:
        for i in range(self.popSize):
            gbest = self.swarm[self.gBestIndex]
            p = self.swarm[i]
            if rand(0,1) < self.pns:
                # neighbourhood search
                r = [rand(0,1) for _ in range(6)]
                sum1 = r[0] + r[1] + r[2]
                sum2 = r[3] + r[4] + r[5]
                for t in range(3):
                    r[t] = r[t] / sum1
                    r[t + 3] = r[t + 3] / sum2
                # LNS
                # choose two different neighbours
                ln = sample([(i + n) % self.popSize for n in range(-self.k, self.k + 1) if n != 0], k = 2)
                pc = self.swarm[ln[0]]
                pd = self.swarm[ln[1]]
                lposition = [0.0 for _ in range(self.dim)]
                for d in range(self.dim):
                    lposition[d] = r[0] * p.x[d] + r[1] * p.pbest[d] \
                                    + r[2] * (pc.x[d] - pd.x[d])
                    lposition[d] = max(self.lb[d], min(self.ub[d], lposition[d]))
                # GNS
                # choose two different neighbours
                gn = sample([n for n in range(self.popSize) if n != i], k = 2)
                pe = self.swarm[gn[0]]
                pf = self.swarm[gn[1]]
                gposition = [0.0 for _ in range(self.dim)]
                for d in range(self.dim):
                    gposition[d] = r[3] * p.x[d] + r[4] * gbest.pbest[d] \
                                    + r[5] * (pe.x[d] - pf.x[d])
                    gposition[d] = max(self.lb[d], min(self.ub[d], gposition[d]))
                # evaluate and exchange
                flp = self.f(lposition)
                fgp = self.f(gposition)
                if self.fitter(flp, p.fx):
                    p.x = [d for d in lposition]
                    p.fx = flp
                if self.fitter(fgp, p.fx):
                    p.x = [d for d in gposition]
                    p.fx = fgp
                # update pbest and gbest
                if self.fitter(p.fx, p.fpbest):
                    p.updatePbest()
                    if self.fitter(p.fpbest, gbest.fpbest):
                        self.gBestIndex = i

    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        if self.fecounter in self.samplePoints:
            self.result.append((self.fecounter, self.g, self.err(self.swarm[self.gBestIndex].fpbest)))
        return self.evaluate(x)