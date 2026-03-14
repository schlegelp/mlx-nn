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
    for D in [3, 10, 50]:
        for N in [100, 1000, 10000, 50000]:
            for k in [1, 5, 10]:
                np.random.seed(0)
                data = np.random.randint(0, 1000, size=(N, D)).astype(np.float32)
                x = np.random.randint(0, 1000, size=(N, D)).astype(np.float32)
                print(f"D = {D}; N = {N}; k = {k}")
                func = partial(KDTree, data)
                time_build = time_func(
                    func, repeats=max(1, min(100, int(1000000 / D / N) * 2))
                )
                print(
                    f"  KDTree - Build time: {np.mean(time_build):<10.6f} (+/- {np.std(time_build):<10.6f})"
                )
                tree = KDTree(data)
                func = partial(tree.query, x, k=k)
                timings_query_tree = time_func(
                    func, repeats=max(1, min(100, int(1000000 / D / N) * 2))
                )
                print(
                    f"  KDTree - Query time: {np.mean(timings_query_tree):<10.6f} (+/- {np.std(timings_query_tree):<10.6f})"
                )
                print("mlx_nn")
                nn = mnn.KNN(data)
                mx.eval(nn._data_norms)  # Make sure this finished before timing
                x = mx.array(x)
                func = partial(nn.query, x, k=k)
                timings_query_nn = time_func(
                    func, repeats=max(1, min(100, int(1000000 / D / N) * 2))
                )
                print(
                    f"  mlx-nn - Query time: {np.mean(timings_query_nn):<10.6f} (+/- {np.std(timings_query_nn):<10.6f})"
                )
                print("\n")
                results.append(
                    [
                        D,
                        N,
                        k,
                        np.mean(time_build),
                        np.std(time_build),
                        np.mean(timings_query_tree),
                        np.std(timings_query_tree),
                        np.mean(timings_query_nn),
                        np.std(timings_query_nn),
                    ]
                )
    df = pd.DataFrame(
        results,
        columns=[
            "D",
            "N",
            "k",
            "pykdtree build",
            "pykdtree build std",
            "pykdtree query",
            "pykdtree std",
            "mlx-nn query",
            "mlx-nn std",
        ],
    )

    # Add a column for the speedup factor
    df["speedup"] = df["pykdtree query"] / df["mlx-nn query"]

    print(
        df[["D", "N", "k", "pykdtree query", "mlx-nn query", "speedup"]].to_markdown()
    )
