"""common.py common constant and data structure
"""

class BaseParticle:
    def __init__(self, D) -> None:
        self.x = [0.0 for _ in range(D)]
        self.pbest = [0.0 for _ in range(D)]
        self.fx:float = None
        self.fpbest:float = None
    
    def updatePbest(self):
        """
        set current x as new pbest
        """
        self.pbest = [i for i in self.x]
        self.fpbest = self.fx