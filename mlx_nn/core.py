import numpy as np
import mlx.core as mx

from functools import lru_cache

__all__ = ["KNN"]


class KNN:
    """K-Nearest Neighbor look-up using brute-force search on the GPU.

    Parameters
    ----------
    data :   (N, M) array_like
                The set of points to search. Must be a 2D numpy or
                mlx array with N points in M dimensions.
    """

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            data = mx.array(data)  # Automatically converts to float32
        elif not isinstance(data, mx.array):
            raise TypeError("`data` must be numpy array or mlx array")

        if data.ndim != 2:
            raise ValueError("`data` must be a 2D array")

        self.data = data
        # Precompute norms (||y||^2) for each data point for fast matmul-based kNN.
        self._data_norms = mx.sum(data * data, axis=1)
        mx.eval(self._data_norms)  # Pre-evaluate norms to avoid GPU synchronization during querys

    def __repr__(self):
        return f"KNN(data.shape={self.data.shape})"

    def query(self, x, k=1, return_numpy=True):
        """Find the k nearest neighbors.

        Parameters
        ----------
        x :     (N, M) array_like
                The point to search. Must be a 2D numpy or mlx array
                with N points in M dimensions.
        k :     int
                The number of nearest neighbors to return. Must be a
                positive integer.
        return_numpy : bool, optional
                If False, the returned arrays will be mlx arrays which
                may be returned before the GPU computation is complete.
                It's the caller's responsibility that they are evaluated
                before further use (e.g. via `mx.eval`).

        Returns
        -------
        dist :  (N, k) array_like (see `return_numpy` parameter)
                The distances of the k nearest neighbors to x in data.
        index : (N, k) array_like (see `return_numpy` parameter)
                The index of the k nearest neighbors to x in data.

        """
        if isinstance(x, np.ndarray):
            x = mx.array(x)  # Automatically converts to float32
        elif not isinstance(x, mx.array):
            raise TypeError("`x` must be numpy array or mlx array")

        if x.ndim != 2:
            raise ValueError("`x` must be a 2D array")

        # Precompute query norms (||x||^2) once.
        x_norms = mx.sum(x * x, axis=1)

        # Initialize best-k buffers on the GPU.
        N = x.shape[0]
        best_dists = mx.full((N, k), np.finfo(np.float32).max, dtype=x.dtype)
        best_indices = mx.zeros((N, k), dtype=mx.int32)

        kernel = _build_blocked_matmul_topk_kernel(k)

        # Process the dataset in blocks to keep memory usage bounded.
        # Larger blocks tend to be faster but use more GPU memory.
        block_size = 256
        M = self.data.shape[0]
        for block_start in range(0, M, block_size):
            block_end = min(M, block_start + block_size)
            block = self.data[block_start:block_end]

            block_norms = self._data_norms[block_start:block_end]
            mat = mx.matmul(x, block.T)

            best_indices, best_dists = kernel(
                inputs=[
                    x_norms,
                    block_norms,
                    mat,
                    best_dists,
                    best_indices,
                    np.uint32(block_start),
                ],
                grid=(N, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(N, k), (N, k)],
                output_dtypes=[mx.int32, x.dtype],
            )

        # Converting to numpy arrays will evaluate the mx arrays
        if not return_numpy:
            return best_dists, best_indices
        else:
            return np.array(best_dists, copy=False), np.array(best_indices, copy=False)



# We're keeping this old kernel around for benchmarking and testing purposes,
# but it's not used in the main KNN implementation anymore since the
# blocked matmul approach is much faster.
@lru_cache
def _build_pairwise_closest_kernel(k):
    source = r"""
    uint i = thread_position_in_grid.x;
    uint N_ = inp0_shape[0];
    uint M_ = inp1_shape[0];
    uint D_ = inp0_shape[1];
    if (i >= N_) return;

    float best_dists[{k}];
    uint best_indices[{k}];

    // Initialize with max float
    for (uint ki = 0; ki < {k}; ++ki) {{
        best_dists[ki] = 3.4e38f;
        best_indices[ki] = 0;
    }}

    uint D4 = D_ / 4;

    for (uint j = 0; j < M_; ++j) {{
        float d2 = 0.0f;
        uint base0 = i * D_;
        uint base1 = j * D_;

        // Vectorized 4-element dot-product loop using float4 and dot()
        for (uint dim = 0; dim < D4 * 4; dim += 4) {{
            float4 diff = float4(
                inp0[base0 + dim] - inp1[base1 + dim],
                inp0[base0 + dim + 1] - inp1[base1 + dim + 1],
                inp0[base0 + dim + 2] - inp1[base1 + dim + 2],
                inp0[base0 + dim + 3] - inp1[base1 + dim + 3]
            );
            d2 += dot(diff, diff);
        }}

        // Remainder dims (if D_ not multiple of 4)
        for (uint dim = D4 * 4; dim < D_; ++dim) {{
            float diff = inp0[base0 + dim] - inp1[base1 + dim];
            d2 = fma(diff, diff, d2);
        }}

        // Insert into sorted list if needed
        for (uint ki = 0; ki < {k}; ++ki) {{
            if (d2 < best_dists[ki]) {{
                // Shift elements to make room
                for (uint kj = {k} - 1; kj > ki; --kj) {{
                    best_dists[kj] = best_dists[kj - 1];
                    best_indices[kj] = best_indices[kj - 1];
                }}
                best_dists[ki] = d2;
                best_indices[ki] = j;
                break;
            }}
        }}
    }}

    // Write outputs
    for (uint ki = 0; ki < {k}; ++ki) {{
        out0[i * {k} + ki] = best_indices[ki];
        out1[i * {k} + ki] = best_dists[ki];
    }}
    """.format(
        k=k
    )
    return mx.fast.metal_kernel(
        name=f"pairwise_closest_{k}d",
        input_names=["inp0", "inp1"],
        output_names=["out0", "out1"],
        source=source,
    )


@lru_cache
def _build_blocked_matmul_topk_kernel(k):
    source = r"""
    uint i = thread_position_in_grid.x;
    uint N_ = x_norms_shape[0];
    uint B_ = mat_shape[1];
    if (i >= N_) return;

    float xn = x_norms[i];

    float best_dists[{k}];
    uint best_indices[{k}];

    // Initialize from previous best values
    for (uint ki = 0; ki < {k}; ++ki) {{
        best_dists[ki] = best_dists_in[i * {k} + ki];
        best_indices[ki] = best_indices_in[i * {k} + ki];
    }}

    for (uint j = 0; j < B_; ++j) {{
        float m = mat[i * B_ + j];
        float d2 = xn + block_norms[j] - 2.0f * m;

        // Merge into best-k list
        for (uint ki = 0; ki < {k}; ++ki) {{
            if (d2 < best_dists[ki]) {{
                for (uint kj = {k} - 1; kj > ki; --kj) {{
                    best_dists[kj] = best_dists[kj - 1];
                    best_indices[kj] = best_indices[kj - 1];
                }}
                best_dists[ki] = d2;
                best_indices[ki] = block_offset + j;
                break;
            }}
        }}
    }}

    for (uint ki = 0; ki < {k}; ++ki) {{
        out0[i * {k} + ki] = best_indices[ki];
        out1[i * {k} + ki] = best_dists[ki];
    }}
    """.format(k=k)
    return mx.fast.metal_kernel(
        name=f"blocked_matmul_topk_{k}d",
        input_names=[
            "x_norms",
            "block_norms",
            "mat",
            "best_dists_in",
            "best_indices_in",
            "block_offset",
        ],
        output_names=["out0", "out1"],
        source=source,
    )
