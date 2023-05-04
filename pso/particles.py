"""particles.py definition of different particle structure
"""

class BasicParticle:
    def __init__(self, D:int) -> None:
        self.v = [0.0 for _ in range(D)]
        self.x = [0.0 for _ in range(D)]
        self.pbest = [0.0 for _ in range(D)]
        self.xVal:float = None
        self.pbVal:float = None

    def updatePbest(self):
        self.pbest = [i for i in self.x]
        self.pbVal = self.xVal