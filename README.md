# mlx-nn
Brute force nearest-neighbour lookup for Apple Silicon.

This is mostly an experiment with my shiny new M3 Pro MacBook. Thanks to the "unified"
(shared) memory we should be able to move data to/from the GPU with near-zero overhead.
All of that is made easy by the [mlx](https://ml-explore.github.io/mlx/build/html/index.html)
framework which allows us to write custom Metal Kernels.

Here, we're using a brute-force approach for KNN search on the GPU.

Long story short: I thought I'd give it a shot.

Below table shows a comparison against the KD-tree implementation from
[pykdtree](https://github.com/storpipfugl/pykdtree) querying `(N, D)` points against `(N, D)` data, where `N` is the number of points and `D` is the dimensionality.

|    |   D |     N |   k |   pykdtree query |   mlx-nn query |    speedup |
|---:|----:|------:|----:|-----------------:|---------------:|-----------:|
|  0 |   3 |   100 |   1 |      4.99344e-05 |    0.000267274 |  0.186828  |
|  1 |   3 |   100 |   5 |      5.16343e-05 |    0.000327551 |  0.157637  |
|  2 |   3 |   100 |  10 |      5.22518e-05 |    0.000338101 |  0.154545  |
|  3 |   3 |  1000 |   1 |      5.9433e-05  |    0.000315356 |  0.188463  |
|  4 |   3 |  1000 |   5 |      9.08804e-05 |    0.000662735 |  0.137129  |
|  5 |   3 |  1000 |  10 |      0.000114498 |    0.00105174  |  0.108865  |
|  6 |   3 | 10000 |   1 |      0.000523249 |    0.0037955   |  0.13786   |
|  7 |   3 | 10000 |   5 |      0.000887029 |    0.00505393  |  0.175513  |
|  8 |   3 | 10000 |  10 |      0.00127389  |    0.00781827  |  0.162937  |
|  9 |   3 | 50000 |   1 |      0.00348874  |    0.0828297   |  0.0421194 |
| 10 |   3 | 50000 |   5 |      0.00491935  |    0.0923172   |  0.0532874 |
| 11 |   3 | 50000 |  10 |      0.00627802  |    0.12187     |  0.0515139 |
| 12 |  10 |   100 |   1 |      5.16486e-05 |    0.000239425 |  0.21572   |
| 13 |  10 |   100 |   5 |      5.14579e-05 |    0.000300581 |  0.171194  |
| 14 |  10 |   100 |  10 |      5.67317e-05 |    0.000357983 |  0.158476  |
| 15 |  10 |  1000 |   1 |      0.000232437 |    0.000383937 |  0.605403  |
| 16 |  10 |  1000 |   5 |      0.000365469 |    0.000780497 |  0.468252  |
| 17 |  10 |  1000 |  10 |      0.000526056 |    0.00117096  |  0.449251  |
| 18 |  10 | 10000 |   1 |      0.0063022   |    0.00412508  |  1.52778   |
| 19 |  10 | 10000 |   5 |      0.0127692   |    0.00565701  |  2.25724   |
| 20 |  10 | 10000 |  10 |      0.018895    |    0.00807678  |  2.33942   |
| 21 |  10 | 50000 |   1 |      0.048198    |    0.0971638   |  0.496049  |
| 22 |  10 | 50000 |   5 |      0.0878305   |    0.0943261   |  0.931136  |
| 23 |  10 | 50000 |  10 |      0.120586    |    0.122602    |  0.983556  |
| 24 |  50 |   100 |   1 |      0.000176418 |    0.000249367 |  0.707463  |
| 25 |  50 |   100 |   5 |      0.000200193 |    0.000312343 |  0.64094   |
| 26 |  50 |   100 |  10 |      0.000223842 |    0.000333762 |  0.670662  |
| 27 |  50 |  1000 |   1 |      0.00195239  |    0.000556582 |  3.50781   |
| 28 |  50 |  1000 |   5 |      0.002019    |    0.00100505  |  2.00885   |
| 29 |  50 |  1000 |  10 |      0.00195149  |    0.00138302  |  1.41104   |
| 30 |  50 | 10000 |   1 |      0.154791    |    0.00439399  | 35.2279    |
| 31 |  50 | 10000 |   5 |      0.167584    |    0.00577801  | 29.0038    |
| 32 |  50 | 10000 |  10 |      0.159174    |    0.00857425  | 18.5642    |
| 33 |  50 | 50000 |   1 |      4.05107     |    0.0930214   | 43.5499    |
| 34 |  50 | 50000 |   5 |      4.0184      |    0.0961139   | 41.8088    |
| 35 |  50 | 50000 |  10 |      3.99134     |    0.129979    | 30.7076    |

*Speed-up is calculated as `pykdtree query / mlx-nn query`, i.e. `<1` means `pykdtree` is faster and `>1` means `mlx-nn` is faster.
Note that these are just the query times. Build times for the KD-tree are not shown here but they are typically negiligible compared to the query times, especially as `N` grows.

As you can see `pykdtree` is much faster in low-dimensional space (I think
it may be optimised for that) but `mlx-nn` gets the upper hand as the computations
become denser in higher dimensions.

Unfortunately, I'm working with low-dimensional data so this doesn't really help me personally. I might still tinker around with it a bit more - there
are more clever implementations of KNN searches on GPUs (see e.g.
[torch_cluster.knn](https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/knn.py)).
