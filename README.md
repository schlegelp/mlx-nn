# mlx-nn 
Brute force nearest-neighbour lookup for Apple silicon.

This is mostly an experiment with my shiny new M3 Pro MacBook. Thanks to the "unified"
(shared) memory we should be able to move data to/from the GPU with near-zero overhead.
All of that is made easy by the [mlx](https://ml-explore.github.io/mlx/build/html/index.html)
framework which even compiles functions into a computation graph for us.

Long story short: I thought I'd give it a shot.

Below table shows a comparison against 
[pykdtree](https://github.com/storpipfugl/pykdtree) querying `(N, D)` points:

|    D |     N |   mlx_nn mean |   mlx_nn std |   KDTree mean |   KDTree std |
|----:|------:|--------------:|-------------:|--------------:|-------------:|
|    3 |   100 |   0.000344038 |  0.000468124 |   1.15395e-05 |  3.16693e-06 |
|      |  1000 |   0.00106812  |  0.000557395 |   0.000294685 |  1.50529e-05 |
|      | 10000 |   0.0100892   |  0.00279511  |   0.00241501  |  0.000127968 |
|   10 |   100 |   0.000200033 |  7.14507e-05 |   4.90904e-05 |  2.24532e-06 |
|      |  1000 |   0.000262022 |  9.54015e-05 |   0.00257761  |  6.36754e-05 |
|      | 10000 |   0.0101402   |  9.7066e-05  |   0.0743439   |  0.000517674 |
|   50 |   100 |   0.000194073 |  3.16612e-05 |   0.000196791 |  1.35794e-06 |
|     |  1000 |   0.000295877 |  0.000329374 |   0.0223814   |  0.00037478  |
|     | 10000 |   0.0103729   |  0.00224273  |   2.37037     |  0.0311136   |
|  100 |   100 |   0.000218153 |  1.16761e-05 |   0.000471711 |  5.04665e-06 |
|    |  1000 |   0.000812054 |  0.000241365 |   0.052322    |  3.72401e-05 |
|    | 10000 |   0.0116751   |  0.00198946  |   5.42449     |  0.0654594   |

As you can see `pykdtree` is much faster in low-dimensional space (I think 
it may be optimised for that) but `mlx-nn` becomes faster as the computations
become denser.

Unfortunately, I'm working with low-dimensional data so this doesn't really help me personally. I might still tinker around with it a bit more - there 
are more clever implementations of KNN searches on GPUs (see e.g. 
[torch_cluster.knn](https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/knn.py)).
