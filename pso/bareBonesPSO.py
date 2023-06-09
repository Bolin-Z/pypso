"""bareBonesPSO.py Bare Bones Particle Swarm 
"""
from .common import BaseParticle
from functions.problem import Problem
from random import gauss as gauss
from random import uniform as rand

class BareBonesParticle(BaseParticle):
    def __init__(self, D:int) -> None:
        super().__init__(D)

class BareBonesPSO:
    def run(self) -> tuple[float, list[float]]:
        while self.g < self.G:
            self._updateSwarm()
            self._updateGbest()
            self.g += 1
        gbest = self.swarm[self.gBestIndex]
        return (gbest.fpbest, gbest.pbest)

    def __init__(
            self,
            objectFunction:Problem,
            populationSize:int = 20,
            maxGeneration:int = 4000,
            interactionProbability:float = 0.5,
            initialSwarm:list[BareBonesParticle] = None
        ) -> None:
        
        self.f = objectFunction.evaluate
        self.fitter = objectFunction.fitter

        self.dim = objectFunction.D
        self.popSize = populationSize
        self.G = maxGeneration
        self.ip = interactionProbability
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
            newParticle = BareBonesParticle(self.dim)
            for d in range(self.dim):
                newParticle.x[d] = rand(self.lb[d], self.ub[d])
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
                # interaction probability
                r = rand(0,1)
                if r < 0.5:
                    # use gauss distribution to update
                    p.x[d] = gauss(
                        mu = (p.pbest[d] + gBest.pbest[d]) / 2,
                        sigma = (abs(p.pbest[d] - gBest.pbest[d])) 
                    )
                else:
                    # learn from previous best
                    p.x[d] = p.pbest[d]
                # amend position
                p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            # evaluate fitness and update pbest
            p.fx = self.f(p.x)
            if (self.fitter(p.fx, p.fpbest)):
                p.updatePbest()