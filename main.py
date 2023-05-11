from pso import *
from copy import deepcopy
from functions import *
from numpy import mean, median, std
import os

if __name__ == "__main__":
    # lab parameters
    dimension = 10
    runNumber = 50
    sampleFEsPoints:list[float] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # define benchmark functions
    problems = {
        "Sphere" : Sphere(
            [-5.12 for _ in range(dimension)],
            [ 5.12 for _ in range(dimension)]
        ),
        "SumSquare" : SumSquare(
            [-5.12 for _ in range(dimension)],
            [ 5.12 for _ in range(dimension)]
        ),
        "schaffer" : Schaffer(
            [-100 for _ in range(2)],
            [ 100 for _ in range(2)]
        ),
        "Ackley" :  Ackley(
            [-32.768 for _ in range(dimension)],
            [ 32.768 for _ in range(dimension)]
        ),
        "Rastrigin" : Rastrigin(
            [-5.12 for _ in range(dimension)],
            [ 5.12 for _ in range(dimension)]
        ),
        "Griewand" : Griewand(
            [-600 for _ in range(dimension)],
            [ 600 for _ in range(dimension)]
        ),
        "Rosenbrock" : Rosenbrock(
            [-5 for _ in range(dimension)],
            [10 for _ in range(dimension)]
        ),
        "Zakharov" : Zakharov(
            [-5 for _ in range(dimension)],
            [10 for _ in range(dimension)]
        ),
        "Easom" : Easom(
            [-100 for _ in range(dimension)],
            [ 100 for _ in range(dimension)]
        ),
        "Step" : Step(
            [-100 for _ in range(dimension)],
            [ 100 for _ in range(dimension)]
        ),
        "QuarticWithNoise" : QuarticWithNoise(
            [-1.28 for _ in range(dimension)],
            [ 100 for _ in range(dimension)]
        )
    }
    # tested algorithms
    algs = [
        OriginalPSO,
        CanonicalPSO,
        BareBonesPSO,
        AIWPSO,
        ALCPSO,
        VonNeumannPSO,
        DMSPSO,
        OLPSO,
        EPSO,
        ASDPSO,
        SAPSOMVS,
        RVUPSO,
        DNSPSO,
        APSO,
        FDRPSO,
        CLPSO
    ]

    for alg in algs:
        dirPath = "./result/" + alg.__name__
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        print("#" * 40)
        print(f"Alg:{alg.__name__}")
        for func in problems:
            filePath = dirPath + "/" + func + ".txt"
            with open(filePath, 'w', encoding="UTF-8") as f:
                f.write(f"{func}\n\n")
                print("-" * 30)
                print(f"start\nfunc:{func}")
                results = []
                for i in range(runNumber):
                    print(f"+ test {i}")
                    f.write(f"test {i}\n")
                    pso = alg(objectFunction = problems[func], samplePoints = sampleFEsPoints, maxFEs = problems[func].D * 10000)
                    res = pso.run()
                    results.append(res)
                    f.write("FEs\t\tGen\t\tErr\n")
                    for t in res:
                        fe, g, err = t
                        f.write(f"{fe}\t\t{g}\t\t{err}\n")
                    f.write("\n\n")
                    print(f"- finish test {i}")
                # analysis
                print("start analysis")
                recordlen = len(results[0])
                finalSol = [results[i][recordlen - 1][2] for i in range(runNumber)]
                finalSol.sort()
                bestVal= finalSol[0]
                worstVal = finalSol[-1]
                meanVal = mean(finalSol)
                medianVal = median(finalSol)
                stdevVal = std(finalSol)
                f.write(f"---------- Analysist ----------\n")
                f.write(f"Runs:{runNumber}\n")
                f.write("Best: %.4E\nWorst: %.4E\nMedian: %.4E\n" % (bestVal, worstVal, medianVal))
                f.write("Mean(SD): %.4E(%.4E)\n\n" % (meanVal, stdevVal))
                # compute the mean of each FEs
                f.write("Mean of Err in each FEs\n")
                f.write("FEs\t\tMean(SD)\n")
                for i in range(recordlen):
                    solutions = [results[x][i][2] for x in range(runNumber)]
                    f.write("%d\t\t%.4E(%.4E)\n" % (results[0][i][0], mean(solutions), std(solutions)))
                print("finish analysis")
                    
