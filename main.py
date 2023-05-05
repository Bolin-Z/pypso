from random import random
from pso.canonicalPSO import CanonicalPSO
from pso.bareBonesPSO import BareBonesPSO
from functions import *


if __name__ == "__main__":
    fit = Schaffer(
        [-100 for _ in range(2)],
        [ 100 for _ in range(2)]
    )
    pso1 = CanonicalPSO(fit)
    val, res = pso1.run()
    print(val)
    print(res)
    pso2 = BareBonesPSO(fit)
    val, res = pso2.run()
    print(val)
    print(res)