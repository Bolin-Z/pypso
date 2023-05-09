"""DMSPSO.py dynamic multi-swarm PSO
"""
from .vonNeumannPSO import ParticleWithNeighbours
from functions.problem import Problem
from random import uniform as rand
from random import shuffle as shuffle

class DMSPSO:
    def run(self) -> tuple[float, list[float]]:
        globalMode = False
        while self.g < self.G:
            self._updateSwarm(globalMode)
            self._updateGbest()
            self._updateInertiaWeight()
            self.g += 1
            if not globalMode and self.g % self.R == 0:
                self._regroup()
            if not globalMode and self.g >= 0.9 * self.G:
                globalMode = True
        gbest = self.swarm[self.gBestIndex]
        return (gbest.fpbest, gbest.pbest)

    def __init__(
            self,
            objectFunction:Problem,
            subSwarmSize:int = 3,
            subSwarmNumber:int = 10,
            regroupPeriod:int = 5,
            maxGeneration:int = 2000,
            c1:float = 1.49445,
            c2:float = 1.49445,
            wmin:float = 0.2,
            wmax:float = 0.9,
            vmaxPercent:float = 0.2,
            initialSwarm:list[ParticleWithNeighbours] = None
        ) -> None:
        
        self.f = objectFunction.evaluate
        self.fitter = objectFunction.fitter

        self.dim = objectFunction.D
        self.subSwarmSize = subSwarmSize
        self.subSwarmNumber = subSwarmNumber
        self.popSize = self.subSwarmSize * self.subSwarmNumber
        self.R = regroupPeriod
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
        self._regroup()

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

    def _regroup(self) -> None:
        permutation = [x for x in range(self.popSize)]
        shuffle(permutation)
        for subSwarmIdx in range(0, self.popSize, self.subSwarmSize):
            for offset in range(self.subSwarmSize):
                curIdx = subSwarmIdx + offset
                curP = self.swarm[permutation[curIdx]]
                curP.neighbours.clear()
                for neighbor in range(self.subSwarmSize):
                    neighborIdx = subSwarmIdx + neighbor
                    if neighborIdx != curIdx:
                        curP.neighbours.append(self.swarm[permutation[neighborIdx]])
    
    def _updateGbest(self) -> None:
        for i in range(self.popSize):
            if self.fitter(self.swarm[i].fpbest, self.swarm[self.gBestIndex].fpbest):
                self.gBestIndex = i

    def _updateSwarm(self, globalMode:bool) -> None:
        for i in range(self.popSize):
            p = self.swarm[i]
            exemplar = self.swarm[self.gBestIndex] if globalMode else self._getLocalBest(i)
            for d in range(self.dim):
                # update velocity
                p.v[d] = self.w * p.v[d] + self.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                            + self.c2 * rand(0,1) * (exemplar.pbest[d] - p.x[d])
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