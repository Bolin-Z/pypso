from random import random
from pso.canonicalPSO import CanonicalPSO
from functions import *


if __name__ == "__main__":
    fit = Schaffer(
        [-100 for _ in range(2)],
        [ 100 for _ in range(2)]
    )
    pso = CanonicalPSO(fit)
    val, res = pso.run()

    print(val)
    print(res)
