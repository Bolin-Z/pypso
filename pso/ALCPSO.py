"""ALCPSO.py Aging Leader and Challengers Particle Swarm Optimization
"""
from .canonicalPSO import CanonicalParticle
from functions.problem import Problem
from random import uniform as rand
from random import randrange as randrange

class Leader:
    def __init__(self, D:int) -> None:
        self.x = [0.0 for _ in range(D)]
        self.fx:float = None
        self.lifeSpan:int = None
        self.age:int = None
    
    def setCurrentPositionAsLeader(self, p:CanonicalParticle) -> None:
        self.x = [x for x in p.x]
        self.fx = p.fx

class LifeSpanController:
    def __init__(self) -> None:
        self.improveGbest:bool = False
        self.improvePbest:bool = False
        self.improveLeader:bool = False
    
    def reset(self) -> None:
        self.improveGbest = False
        self.improvePbest = False
        self.improveLeader = False
    
    def adjuctLifeSpan(self, l:Leader) -> None:
        if self.improveGbest:
            l.lifeSpan += 2
        else:
            if self.improvePbest:
                l.lifeSpan += 1
            else:
                if not self.improveLeader:
                    l.lifeSpan -= 1

class ALCPSO:
    def run(self) -> list[tuple[int, int, float]]:
        while self.fecounter < self.maxFEs:
            self.lsc.reset()
            self._updateSwarm()
            self.lsc.adjuctLifeSpan(self.leader)
            self.leader.age += 1
            if self.leader.age >= self.leader.lifeSpan:
                self._challenging()
            self.g += 1
        return self.result

    def __init__(
            self,
            objectFunction:Problem,
            samplePoints:list[float],
            populationSize:int = 20,
            maxGeneration:int = 4000,
            maxFEs:int = 10000,
            c1:float = 2.0,
            c2:float = 2.0,
            w:float = 0.4,
            initialLifeSpan:int = 60,
            challengeTmax:int = 2,
            vmaxPercent:float = 0.5,
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
        self.g = 0
        self.initialLifeSpan = initialLifeSpan
        self.T = challengeTmax
        self.pro = (1 / self.dim)

        self.lb = objectFunction.lb
        self.ub = objectFunction.ub
        self.vmax = [vmaxPercent * (self.ub[x] - self.lb[x]) for x in range(self.dim)]

        self.swarm = initialSwarm
        if not self.swarm:
            self._initialSwarm()
        
        # find gBest
        self.gBestIndex:int = 0
        for i in range(self.popSize):
            if self.fitter(self.swarm[i].fpbest, self.swarm[self.gBestIndex].fpbest):
                self.gBestIndex = i
        
        # set leader
        self.leader = Leader(self.dim)
        self.leader.age = 0
        self.leader.lifeSpan = self.initialLifeSpan
        self.leader.setCurrentPositionAsLeader(self.swarm[self.gBestIndex])
        # lifespan controller
        self.lsc = LifeSpanController()

    def _initialSwarm(self) -> None:
        self.swarm = []
        for _ in range(self.popSize):
            newParticle = CanonicalParticle(self.dim)
            for d in range(self.dim):
                newParticle.x[d] = rand(self.lb[d], self.ub[d])
            newParticle.fx = self.f(newParticle.x)
            newParticle.updatePbest()
            self.swarm.append(newParticle)

    def _updateSwarm(self) -> None:
        for i in range(self.popSize):
            p =self.swarm[i]
            for d in range(self.dim):
                # update velocity
                p.v[d] = self.w * p.v[d] + self.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                            + self.c2 * rand(0,1) * (self.leader.x[d] - p.x[d])
                p.v[d] = max(-self.vmax[d], min(self.vmax[d], p.v[d]))
                # update position
                p.x[d] = p.x[d] + p.v[d]
                p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
            # evaluate fitness and update pbest leader
            p.fx = self.f(p.x)
            if(self.fitter(p.fx, self.leader.fx)):
                self.leader.setCurrentPositionAsLeader(p)
                self.lsc.improveLeader = True
            if(self.fitter(p.fx, p.fpbest)):
                p.updatePbest()
                self.lsc.improvePbest = True
                # update gbest
                if(self.fitter(p.fx, self.swarm[self.gBestIndex].fpbest)):
                    self.gBestIndex = i
                    self.lsc.improveGbest = True

    def _challenging(self) -> None:
        # record the current status
        positions = [[i for i in p.x] for p in self.swarm]
        velocities = [[i for i in p.v] for p in self.swarm]
        fvals = [p.fx for p in self.swarm]        
        # evaluate the challenger
        challenger = self._generateChallenger()
        for t in  range(self.T):
            improvePbest = False
            for i in range(self.popSize):
                p = self.swarm[i]
                for d in range(self.dim):
                    # update velocity
                    p.v[d] = self.w * p.v[d] + self.c1 * rand(0,1) * (p.pbest[d] - p.x[d]) \
                                + self.c2 * rand(0,1) * (challenger.x[d] - p.x[d])
                    p.v[d] = max(-self.vmax[d], min(self.vmax[d], p.v[d]))
                    # update position
                    p.x[d] = p.x[d] + p.v[d]
                    p.x[d] = max(self.lb[d], min(self.ub[d], p.x[d]))
                # evaluate fitness and update pbest challenger
                p.fx = self.f(p.x)
                if(self.fitter(p.fx, challenger.fx)):
                    challenger.setCurrentPositionAsLeader(p)
                if(self.fitter(p.fx, p.fpbest)):
                    p.updatePbest()
                    improvePbest = True
                    # update gbest
                    if self.fitter(p.fx, self.swarm[self.gBestIndex].fpbest):
                        self.gBestIndex = i
            if improvePbest:
                # accept challenger
                self.leader = challenger
                self.g += (t + 1)
                return

        # reject challenger and roll back
        for i in range(self.popSize):
            p = self.swarm[i]
            p.x = [t for t in positions[i]]
            p.v = [t for t in velocities[i]]
            p.fx = fvals[i]
        self.leader.age = self.leader.lifeSpan - 1

    def _generateChallenger(self) -> Leader:
        # generate a challenger
        challenger = Leader(self.dim)
        count = 0
        for d in range(self.dim):
            if rand(0,1) < self.pro:
                challenger.x[d] = rand(self.lb[d], self.ub[d])
                count += 1
            else:
                challenger.x[d] = self.leader.x[d]
        if count == 0:
            mutatedim = randrange(0, self.dim)
            challenger.x[mutatedim] = rand(self.lb[d], self.ub[d])
        challenger.fx = self.f(challenger.x)
        challenger.age = 0
        challenger.lifeSpan = self.initialLifeSpan
        return challenger
    
    def f(self, x:list[float]) -> float:
        self.fecounter += 1
        if self.fecounter in self.samplePoints:
            self.result.append((self.fecounter, self.g, self.err(self.swarm[self.gBestIndex].fpbest)))
        return self.evaluate(x)