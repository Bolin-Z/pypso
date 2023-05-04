from functions.sphere import Sphere
from random import random
from pso.canonicalPSO import CanonicalPSO


if __name__ == "__main__":
    sphere = Sphere(
        [-10 for _ in range(10)],
        [ 10 for _ in range(10)]
    )

    pso = CanonicalPSO(sphere)
    val, res = pso.run()

    print(val)
    print(res)
