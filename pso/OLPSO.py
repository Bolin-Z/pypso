"""OLPSO.py Orthogonal Learning PSO
"""
from pso.canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand, randrange as randrange
from math import ceil, log2, floor

class OLParticle(CanonicalParticle):
    def __init__(self, D: int) -> None:
        super().__init__(D)
        self.guideIdx:list[int] = [0 for _ in range(D)]
        self.stagnate = 0

class OLPSO:
    def run(self) -> tuple[float, list[float]]:
        while self.g < self.G:
            self._updateSwarm()
            self._updateGbest()
            self._updateGuidance()
            self._updateInertiaWeight()
            self.g += 1
        gbest = self.swarm[self.gBestIndex]
        return(gbest.fpbest, gbest.pbest)

    def __init__(
            self,
            objectFunction:Problem,
            populationSize:int = 20,
            maxGeneration:int = 4000,
            c:float = 2.0,
            wmin:float = 0.4,
            wmax:float = 0.9,
            stagnateMax:int = 5,
            vmaxPercent:float = 0.2,
            initialSwarm:list[OLParticle] = None
        ) -> None:

        self.f = objectFunction.evaluate
        self.fitter = objectFunction.fitter

        self.dim = objectFunction.D
        self.popSize = populationSize
        self.G = maxGeneration
        self.c = c
        self.wmin = wmin
        self.wmax = wmax
        self.stMax = stagnateMax
        self.w = wmax
        self.g = 0

        self.lb = objectFunction.lb
        self.ub = objectFunction.ub
        self.vmax = [vmaxPercent * (self.ub[x] - self.lb[x]) for x in range(self.dim)]

        self.swarm = initialSwarm
        if not self.swarm:
            self._initialSwarm()
        
        self.gBestIndex:int = 0
        self._updateGbest()

        self._generateOA(self.dim)
        for i in range(self.popSize):
            self._constructGuidance(i)

    def _initialSwarm(self) -> None:
        self.swarm = []
        for _ in range(self.popSize):
            newParticle = OLParticle(self.dim)
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
            for d in range(self.dim):
                # update velocity
                examplar = self.swarm[p.guideIdx[d]]
                p.v[d] = self.w * p.v[d] + self.c * rand(0,1) * (examplar.pbest[d] - p.x[d])
                p.v[d] = max(-self.vmax[d], min(self.vmax[d], p.v[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            # evaluate fitness and update pbest
            p.fx = self.f(p.x)
            p.stagnate += 1
            if(self.fitter(p.fx, p.fpbest)):
                p.updatePbest()
                p.stagnate = 0
    
    def _updateGuidance(self) -> None:
        for i in range(self.popSize):
            p = self.swarm[i]
            if p.stagnate > self.stMax:
                self._constructGuidance(i)

    def _updateInertiaWeight(self) -> None:
        self.w = self.wmax - (self.wmax - self.wmin) * (self.g / self.G)

    def _constructGuidance(self, idx:int) -> None:
        p = self.swarm[idx]
        p.stagnate = 0
        # find examplar
        examplarIdx = self.gBestIndex
        while examplarIdx == idx:
            examplarIdx = randrange(0, self.popSize)
        examplar = self.swarm[examplarIdx]
        # construct test set
        x:list[list[float]] = [[0.0 for _ in range(self.dim)] for _ in range(len(self._OA))]
        fx = [0.0 for _ in range(len(self._OA))]
        xBest:int = 0
        for j in range(len(self._OA)):
            for d in range(self.dim):
                x[j][d] = p.pbest[d] if self._OA[j][d] == 1 else examplar.pbest[d]
            fx[j] = self.f(x[j])
            if self.fitter(fx[j], fx[xBest]):
                xBest = j
        # calculate xp
        xp = [0.0 for _ in range(self.dim)]
        for d in range(self.dim):
            levelPbest = 0.0
            levelExamplar = 0.0
            for case in self._oneIdx[d]:
                levelPbest = levelPbest + fx[case]
            for case in self._twoIdx[d]:
                levelExamplar = levelExamplar + fx[case]
            levelPbest = levelPbest / len(self._oneIdx)
            levelExamplar = levelExamplar / len(self._twoIdx)
            if self.fitter(levelPbest, levelExamplar):
                xp[d] = p.pbest[d]
                p.guideIdx[d] = idx
            else:
                xp[d] = examplar.pbest[d]
                p.guideIdx[d] = examplarIdx
        fxp = self.f(xp)
        if self.fitter(fx[xBest], fxp):
            for d in range(self.dim):
                if self._OA[xBest][d] == 1:
                    p.guideIdx[d] = idx
                else:
                    p.guideIdx[d] = examplarIdx

    def _generateOA(self, factorNum:int) -> None:
        n = 2 ** (ceil(log2(factorNum + 1)))
        self._OA = [[0 for _ in range(factorNum)] for _ in range(n)]
        for i in range(1, n + 1):
            for j in range(1, factorNum + 1):
                level = 0
                k = j
                mask = n / 2
                while k > 0:
                    if (k % 2) and (((i-1) & int(mask)) != 0):
                        level = (level + 1) % 2
                    k = floor(k / 2)
                    mask = mask / 2
                self._OA[i - 1][j - 1] = level + 1

        self._oneIdx:list[list[int]] = [[] for _ in range(factorNum)]
        self._twoIdx:list[list[int]] = [[] for _ in range(factorNum)]
        for i in range(n):
            for j in range(factorNum):
                if self._OA[i][j] == 1:
                    self._oneIdx[j].append(i)
                else:
                    self._twoIdx[j].append(i)