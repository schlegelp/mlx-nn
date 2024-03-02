import time 
import numpy as np
import pandas as pd
import mlx_nn as mnn
import mlx.core as mx

from functools import partial

from pykdtree.kdtree import KDTree

def time_func(func, repeats=10, warm_ups=1):
    for i in range(warm_ups):
        _ = func()

    timings = []
    for i in range(repeats):
        start = time.time()
        _ = func()
        timings.append(time.time() - start)
    return timings 


if __name__ == "__main__":
    results = []
    for D in [3, 10, 50, 100]:
        print(f"D = {D}")
        for N in [100, 1000, 10000]:
            data = np.random.randint(0, 1000, size=(N, D))
            x = np.random.randint(0, 1000, size=(N, D))
            print(f"N = {N}")
            print("KDTree")
            tree = KDTree(data)
            func = partial(tree.query, x)
            timings1 = time_func(func)
            print(f"  Average time: {np.mean(timings1)} (+/- {np.std(timings1)})")
            print("mlx_nn")
            nn = mnn.KNN(data)
            x = mx.array(x)
            func = partial(nn.query, x)
            timings2 = time_func(func)
            print(f"  Average time: {np.mean(timings2)} (+/- {np.std(timings2)})")
            print("\n")
            results.append([D, N, np.min(timings2), np.std(timings2), np.mean(timings1), np.std(timings1)])
    df = pd.DataFrame(results, columns=["D", "N", "mlx_nn mean", "mlx_nn std", "KDTree mean", "KDTree std", ])
    print(df.to_markdown())
