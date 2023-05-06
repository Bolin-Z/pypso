from random import random
from pso.canonicalPSO import CanonicalPSO
from pso.AIWPSO import AIWPSO
from pso.vonNeumannPSO import VonNeumannPSO
from pso.ALCPSO import ALCPSO
from copy import deepcopy
from functions import *


if __name__ == "__main__":
    fit = Sphere(
        [-100 for _ in range(5)],
        [ 100 for _ in range(5)],
    )
    test = [
        CanonicalPSO, 
        AIWPSO, 
        VonNeumannPSO,
        ALCPSO
    ]
    for t in test:
        pso = t(fit)
        val, res = pso.run()
        print(val)
        print(res)