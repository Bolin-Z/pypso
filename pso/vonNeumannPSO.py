"""vonNeumannPSO.py von Neumann PSO
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand

class ParticleWithNeighbours(CanonicalParticle):
    def __init__(self, D: int) -> None:
        super().__init__(D)
        self.neighbours:list["ParticleWithNeighbours"] = []

class VonNeumannPSO:
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
            row:int = 5,
            col:int = 4,
            maxGeneration:int = 4000,
            c1:float = 1.49445,
            c2:float = 1.49445,
            wmin:float = 0.4,
            wmax:float = 0.9,
            vmaxPercent:float = 0.2,
            initialSwarm:list[ParticleWithNeighbours] = None
        ) -> None:
        
        self.f = objectFunction.evaluate
        self.fitter = objectFunction.fitter

        self.dim = objectFunction.D
        self.popSize = row * col
        self.row = row
        self.col = col
        self.G = maxGeneration
        self.c1 = c1
        self.c2 = c2
        self.wmin = wmin
        self.wmax = wmax
        self.w = 0.9
        self.g = 0

        self.lb = objectFunction.lb
        self.ub = objectFunction.ub
        self.vmax = [vmaxPercent * (self.ub[x] - self.lb[x]) for x in range(self.dim)]

        self.swarm = initialSwarm
        if not self.swarm:
            self._initialSwarm()
        self._constructNeighbourHood()

        self.gBestIndex:int = 0
        self._updateGbest()

    def _initialSwarm(self) -> None:
        self.swarm = []
        for _ in range(self.popSize):
            newParticle = ParticleWithNeighbours(self.dim)
            for d in range(self.dim):
                newParticle.x[d] = rand(self.lb[d], self.ub[d])
                newParticle.v[d] = rand(-self.vmax[d], self.vmax[d])
            newParticle.fx = self.f(newParticle.x)
            newParticle.updatePbest()
            self.swarm.append(newParticle)

    def _constructNeighbourHood(self) -> None:
        for r in range(self.row):
            for c in range(self.col):
                p = self.swarm[r * self.col + c]
                up = (r - 1) % self.row * self.col + c
                down = (r + 1) % self.row * self.col + c
                left = r * self.col + (c - 1) % self.col
                right = r * self.col + (c + 1) % self.col
                for i in [up, down, left, right]:
                    p.neighbours.append(self.swarm[i])


    def _updateGbest(self) -> None:
        for i in range(self.popSize):
            if self.fitter(self.swarm[i].fpbest, self.swarm[self.gBestIndex].fpbest):
                self.gBestIndex = i

    def _updateSwarm(self) -> None:
        for i in range(self.popSize):
            p = self.swarm[i]
            lbest = self._getLocalBest(i)
            for d in range(self.dim):
                # update velocity
                p.v[d] = self.w * p.v[d] + self.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                            + self.c2 * rand(0,1) * (lbest.pbest[d] - p.x[d])
                p.v[d] = max(-self.vmax[d], min(self.vmax[d], p.v[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            # evaluate fitness and update pbest
            p.fx = self.f(p.x)
            if(self.fitter(p.fx, p.fpbest)):
                p.updatePbest()
    
    def _getLocalBest(self, idx:int) -> ParticleWithNeighbours:
        p = self.swarm[idx]
        res = p
        for n in p.neighbours:
            if self.fitter(n.fpbest, res.fpbest):
                res = n
        return res

    def _updateInertiaWeight(self) -> None:
        self.w = self.wmax - (self.wmax - self.wmin) * (self.g / self.G)