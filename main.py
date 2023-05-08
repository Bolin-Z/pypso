from pso import *
from copy import deepcopy
from functions import *

if __name__ == "__main__":
    problems = {
        "sphere" : Sphere(
            [-100 for _ in range(5)],
            [ 100 for _ in range(5)]
        ),
        "schaffer" : Schaffer(
            [-100 for _ in range(2)],
            [ 100 for _ in range(2)]
        )
    }
    algs = [
        # OriginalPSO,
        CanonicalPSO,
        # BareBonesPSO,
        # AIWPSO, 
        # ALCPSO,
        # VonNeumannPSO,
        # DMSPSO,
        # OLPSO,
        # EPSO,
        # ASDPSO,
        # SAPSOMVS,
        RVUPSO
    ]
    for f in problems:
        print(f"{f}:")
        for alg in algs:
            pso = alg(problems[f])
            val, res = pso.run()
            print(f"\t{alg.__name__}")
            print(f"\t\t{val}")
            print(f"\t\t{res}")