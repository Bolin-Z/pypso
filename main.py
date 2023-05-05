from random import random
from pso.canonicalPSO import CanonicalPSO
from pso.AIWPSO import AIWPSO
from copy import deepcopy
from functions import *


if __name__ == "__main__":
    fit = Sphere(
        [-100 for _ in range(5)],
        [ 100 for _ in range(5)],
    )
    pso1 = CanonicalPSO(fit)
    pso2 = AIWPSO(fit)
    val, res = pso1.run()
    print(val)
    print(res)
    val, res = pso2.run()
    print(val)
    print(res)