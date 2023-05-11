"""ASDPSO.py Adaptive Search Diversification in PSO
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand
from math import sqrt as sqrt

class ASDParticle(CanonicalParticle):
    def __init__(self, D: int) -> None:
        super().__init__(D)
        self.lastx = [0.0 for _ in range(D)]
        self.lastfx:float = None
        self.c1 = [0.0 for _ in range(D)]
        self.c2 = [0.0 for _ in range(D)]
        self.w = [0.0 for _ in range(D)]
    
    def recordLastPosition(self) -> None:
        self.lastx = [i for i in self.x]
        self.lastfx = self.fx

class ASDPSO:
    def run(self) -> list[tuple[int, int, float]]:
        while self.fecounter < self.maxFEs:
            self._calculateParameters()
            self._updateSwarm()
            if self._updateGbest():
                # find new gBest
                p = self.swarm[self.gBestIndex]
                for d in range(self.dim):
                    p.v[d] = 0.0
                    p.x[d] = p.lastx[d] * rand(-0.9, 1.1)
                p.fx = self.f(p.x)
                if self.fitter(p.fx, p.fpbest):
                    p.updatePbest()
            self.g += 1
        return self.result

    def __init__(
            self,
            objectFunction:Problem,
            samplePoints:list[float],
            populationSize:int = 30,
            maxGeneration:int = 4000,
            maxFEs:int = 10000,
            c1max:float = 3.0,
            c2min:float = 0.5,
            c2max:float = 3.0,
            wmin:float = 0.4,
            wmax:float = 0.9,
            initialSwarm:list[ASDParticle] = None
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
        self.c1max = c1max
        self.c2min = c2min
        self.c2max = c2max
        self.win = wmin
        self.wmax = wmax
        self.g = 0

        self.lb = objectFunction.lb
        self.ub = objectFunction.ub

        self.swarm = initialSwarm
        if not self.swarm:
            self._initialSwarm()
        
        self.gBestIndex:int = 0
        self._updateGbest()

    def _initialSwarm(self) -> None:
        self.swarm = []
        for _ in range(self.popSize):
            newParticle = ASDParticle(self.dim)
            for d in range(self.dim):
                newParticle.x[d] = rand(self.lb[d], self.ub[d])
                newParticle.v[d] = 0.0
            newParticle.fx = self.f(newParticle.x)
            newParticle.updatePbest()
            self.swarm.append(newParticle)

    def _updateGbest(self) -> bool:
        lastGbestIdx = self.gBestIndex
        for i in range(self.popSize):
            if self.fitter(self.swarm[i].fpbest, self.swarm[self.gBestIndex].fpbest):
                self.gBestIndex = i
        return self.gBestIndex != lastGbestIdx

    def _updateSwarm(self) -> None:
        gBest = self.swarm[self.gBestIndex]
        for i in range(self.popSize):
            p = self.swarm[i]   
            # update particle
            newPosition = [0.0 for _ in range(self.dim)]
            for d in range(self.dim):
                # update velocity
                r1 = rand(-1,1) if p.c1[d] == self.c1max else rand(0,1)
                p.v[d] = p.w[d] * p.v[d] + p.c1[d] * r1 * (p.pbest[d] - p.x[d]) \
                            + p.c2[d] * rand(0,1) * (gBest.pbest[d] - p.x[d])
                newPosition[d] = p.x[d] + p.v[d]
                # check whether it is out of domain
                while (newPosition[d] < self.lb[d]) or (newPosition[d] > self.ub[d]):
                    p.v[d] = p.v[d] * 0.9 * rand(0,1)
                    newPosition[d] = p.x[d] + p.v[d]
            # check whether it is stuck
            if p.lastfx and abs(p.lastfx - p.fx) <= (10 ** (-10)):
                dis = 0
                for d in range(self.dim):
                    dis += (p.lastx[d] - p.x[d]) ** 2
                dis = sqrt(dis)
                if dis <= (10 ** (-5)):
                    # stuck
                    for d in range(self.dim):
                        newPosition[d] = rand(self.lb[d], self.ub[d])
            # record the new position
            p.recordLastPosition()
            p.x = [x for x in newPosition]
            p.fx = self.f(p.x)
            if (self.fitter(p.fx, p.fpbest)):
                p.updatePbest()

    def _calculateParameters(self) -> None:
        gBest = self.swarm[self.gBestIndex]
        for d in range(self.dim):
            # calculate distances in dimension d
            dis = [0.0 for _ in range(self.popSize)]
            dmax:int = 0
            for i in range(self.popSize):
                p = self.swarm[i]
                dis[i] = abs(p.x[d] - gBest.x[d])
                if dis[i] > dis[dmax]:
                    dmax = i
            # calculate parameter in dimension d
            for i in range(self.popSize):
                p = self.swarm[i]
                # calculate c1 in dimension d
                alpha = 4 * self.c1max / (dis[dmax] ** 2)
                if dis[i] > (dis[dmax] / 2):
                    p.c1[d] = self.c1max
                else:
                    p.c1[d] = alpha * (dis[i] ** 2)
                # calculate c2 in dimension d
                beta = (self.c2max - self.c2min) / (((2/3) * dis[dmax]) ** 2)
                if dis[i] > (dis[dmax] / 3):
                    p.c2[d] = self.c2min + beta * ((dis[dmax] - dis[i]) ** 2)
                else:
                    p.c2[d] = self.c2max
                # calculate w in dimension d
                gamma = (self.wmax - self.win) / (dis[dmax] ** 2)
                p.w[d] = self.win + gamma * (dis[i] ** 2)

    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        if self.fecounter in self.samplePoints:
            self.result.append((self.fecounter, self.g, self.err(self.swarm[self.gBestIndex].fpbest)))
        return self.evaluate(x)
