from pso import *
from copy import deepcopy
from functions import *

if __name__ == "__main__":
    problems = {
        # "Sphere" : Sphere(
        #     [-100 for _ in range(5)],
        #     [ 100 for _ in range(5)]
        # ),
        # "schaffer" : Schaffer(
        #     [-100 for _ in range(2)],
        #     [ 100 for _ in range(2)]
        # ),
        # "Ackley" :  Ackley(
        #     [-32.768 for _ in range(30)],
        #     [ 32.768 for _ in range(30)]
        # ),
        # "Rosenbrock" : Rosenbrock(
        #     [-5 for _ in range(5)],
        #     [10 for _ in range(5)]
        # ),
        "Rastrigin" : Rastrigin(
            [-5.12 for _ in range(5)],
            [ 5.12 for _ in range(5)]
        )
    }
    algs = [
        # OriginalPSO,
        # CanonicalPSO,
        BareBonesPSO,
        # AIWPSO, 
        ALCPSO,
        # VonNeumannPSO,
        # DMSPSO,
        # OLPSO,
        # EPSO,
        # ASDPSO,
        # SAPSOMVS,
        # RVUPSO,
        # DNSPSO,
        APSO
        # FDRPSO,
        # CLPSO
    ]
    for f in problems:
        print(f"{f}:")
        for alg in algs:
            pso = alg(objectFunction = problems[f], maxGeneration = 50000)
            val, res = pso.run()
            print(f"\t{alg.__name__}")
            print(f"\t\t{val}")
            print(f"\t\t{res}")