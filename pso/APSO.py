"""APSO.py Adaptive PSO
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand, gauss, randrange
from math import sqrt, exp

# states constant
_EXPLORATION:int = 0
_EXPLOITATION:int = 1
_CONVERGENCE:int = 2
_JUMPINGOUT:int = 3

class APSO:
    def run(self) -> tuple[float, list[float]]:
        while self.g < self.G:
            self._ESE()
            self._updateParameters()
            self._updateSwarm()
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
            wmin:float = 0.4,
            wmax:float = 0.9,
            elrmin:float = 0.1,
            elrmax:float = 1.0,
            vmaxPercent:float = 0.2,
            initialSwarm:list[CanonicalParticle] = None
        ) -> None:
        
        self.fecounter:int = 0
        self.evaluate = objectFunction.evaluate
        self.fitter = objectFunction.fitter

        self.dim = objectFunction.D
        self.popSize = populationSize
        self.G = maxGeneration
        self.c1 = c1
        self.c2 = c2
        self.wmin = wmin
        self.wmax = wmax
        self.w = self.wmax
        self.elrmin = elrmin
        self.elrmax = elrmax
        self.g = 0

        self.efactor:float = 0.0
        self.state = _EXPLORATION

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
                p.v[d] = self.w * self.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                            + self.c2 * rand(0,1) * (gBest.pbest[d] - p.x[d])
                p.v[d] = max(-self.vmax[d], min(self.vmax[d], p.v[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            # evaluate fitness and update pbest
            p.fx = self.f(p.x)
            if self.fitter(p.fx, p.fpbest):
                p.updatePbest()

    def _ESE(self) -> None:
        # calculate mean distance
        dis = [self._calculateMeanDis(i) for i in range(self.popSize)]
        # calculate evolutionary factor
        dmax = max(dis)
        dmin = min(dis)
        dg = dis[self.gBestIndex]
        self.efactor = (dg - dmin) / (dmax - dmin)
        self._identifyEvolutionState()

    def _updateParameters(self) -> None:
        # update c1 c2
        d1 = rand(0.05, 0.1)
        d2 = rand(0.05, 0.1)
        r1 = rand(0, d1)
        r2 = rand(0, d2)
        if self.state == _EXPLORATION:
            self.c1 = self.c1 + r1
            self.c2 = self.c2 - r2
        elif self.state == _EXPLOITATION:
            self.c1 = self.c1 + 0.5 * r1
            self.c2 = self.c2 - 0.5 * r2
        elif self.state == _CONVERGENCE:
            self.c1 = self.c1 + 0.5 * r1
            self.c2 = self.c2 + 0.5 * r2
            self._ELS()
        elif self.state == _JUMPINGOUT:
            self.c1 = self.c1 - r1
            self.c2 = self.c2 + r2
        self.c1 = max(1.5, min(2.5, self.c1))
        self.c2 = max(1.5, min(2.5, self.c2))
        if not (3.0 <= self.c1 + self.c2 <= 4.0):
            c = self.c1 + self.c2
            self.c1 = self.c1 / c
            self.c2 = self.c2 / c
        # update inertia weight
        self.w = 1 / (1 + 1.5 * exp(-2.6 * self.efactor))
        self.w = max(self.wmin, max(self.w, self.wmax))
    
    def _ELS(self) -> None:
        gbest = self.swarm[self.gBestIndex]
        mutate = randrange(0, self.dim)
        position = [x for x in gbest.pbest]
        elr = self.elrmax - (self.elrmax - self.elrmin) * (self.g / self.G)
        n = gauss(mu = 0, sigma = elr)
        position[mutate] = gbest.pbest[mutate] + (self.ub[mutate] - self.lb[mutate]) * n
        position[mutate] = max(self.lb[mutate], min(position[mutate], self.ub[mutate]))
        fp = self.f(position)
        if self.fitter(fp, gbest.fpbest):
            gbest.pbest =  position
            gbest.fpbest = fp
        else:
            gWorstIndex = 0
            for i in range(self.popSize):
                if self.fitter(self.swarm[gWorstIndex].fpbest, self.swarm[i].fpbest):
                    gWorstIndex = i
            gworst = self.swarm[gWorstIndex]
            if self.fitter(fp, gworst.fpbest):
                gworst.x = position
                gworst.fx = fp
                gworst.updatePbest()

    def _calculateMeanDis(self, idx:int) -> float:
        meandis = 0.0
        p = self.swarm[idx]
        for i in range(self.popSize):
            dis = 0.0
            th =self.swarm[i]
            for d in range(self.dim):
                dis += (p.x[d] - th.x[d]) ** 2
            meandis += sqrt(dis)
        return meandis / (self.popSize - 1)

    def _identifyEvolutionState(self) -> None:
        # compute membership function
        us = [0.0 for _ in range(4)]
        # Exploration
        if 0 <= self.efactor <= 0.4:
            us[_EXPLORATION] = 0.0
        elif 0.4 < self.efactor <= 0.6:
            us[_EXPLORATION] = 5 * self.efactor - 2
        elif 0.6 < self.efactor <= 0.7:
            us[_EXPLORATION] = 1.0
        elif 0.7 < self.efactor <= 0.8:
            us[_EXPLORATION] = -10 * self.efactor + 8
        elif 0.8 < self.efactor <= 1.0:
            us[_EXPLORATION] = 0.0
        # Exploitation
        if 0 <= self.efactor <= 0.2:
            us[_EXPLOITATION] = 0.0
        elif 0.2 < self.efactor <= 0.3:
            us[_EXPLOITATION] = 10 * self.efactor - 2
        elif 0.3 < self.efactor <= 0.4:
            us[_EXPLOITATION] = 1.0
        elif 0.4 < self.efactor <= 0.6:
            us[_EXPLOITATION] = -5 * self.efactor + 3
        elif 0.6 < self.efactor <= 1.0:
            us[_EXPLOITATION] = 0.0
        # Convergence
        if 0 <= self.efactor <= 0.1:
            us[_CONVERGENCE] = 1.0
        elif 0.1 < self.efactor <= 0.3:
            us[_CONVERGENCE] = -5 * self.efactor + 1.5
        elif 0.3 < self.efactor <= 1.0:
            us[_CONVERGENCE] = 0.0
        # Jumping Out
        if 0 <= self.efactor <= 0.7:
            us[_JUMPINGOUT] = 0.0
        elif 0.7 < self.efactor <= 0.9:
            us[_JUMPINGOUT] = 5 * self.efactor - 3.5
        elif 0.9 < self.efactor <= 1:
            us[_JUMPINGOUT] = 1.0
        # classify the state
        zeroCnt = 0
        for u in us:
            if u == 0.0:
                zeroCnt += 1
        if zeroCnt == 3:
            if us[_EXPLORATION] != 0.0:
                self.state = _EXPLORATION
            elif us[_EXPLOITATION] != 0.0:
                self.state = _EXPLOITATION
            elif us[_CONVERGENCE] != 0.0:
                self.state = _CONVERGENCE
            elif us[_JUMPINGOUT] != 0.0:
                self.state = _JUMPINGOUT
        elif zeroCnt == 2:
            if us[_CONVERGENCE] != 0.0 and us[_EXPLOITATION] != 0.0:
                if self.state == _EXPLORATION or self.state == _EXPLOITATION:
                    self.state = _EXPLOITATION
                else:
                    self.state = _CONVERGENCE
            if us[_EXPLOITATION] != 0.0 and us[_EXPLORATION] != 0.0:
                if self.state == _EXPLORATION or self.state == _JUMPINGOUT:
                    self.state = _EXPLORATION
                else:
                    self.state = _EXPLOITATION
            if us[_EXPLORATION] != 0.0 and us[_JUMPINGOUT] != 0.0:
                if self.state == _EXPLORATION or self.state == _JUMPINGOUT:
                    self.state = _EXPLORATION
                else:
                    self.state = _JUMPINGOUT

    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        return self.evaluate(x)
