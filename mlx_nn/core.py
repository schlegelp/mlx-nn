
import mlx.core as mx
import numpy as np

__all__ = ["KNN"]

class KNN:
    """K-Nearest Neighbor Searc
    
    Parameters
    ----------
    data :   (N, M) array_like
                The set of points to search. Must be a 2D numpy or 
                mlx array with N points in M dimensions.    
    """
    def __init__(self, data):       
        if isinstance(data, np.ndarray):
            data = mx.array(data).astype(mx.float32)
        elif not isinstance(data, mx.array):
            raise TypeError("`data` must be numpy array or mlx array")
        
        if data.ndim != 2:
            raise ValueError("`data` must be a 2D array")

        self.data = data

    def query(self, x):
        """Find the nearest neighbor.
        
        Parameters
        ----------
        x :     (N, M) array_like
                The point to search. Must be a 2D numpy or mlx array
                with N points in M dimensions.

        Returns
        -------
        dist :  (N, )
                The distances of the nearest neighbor to x in data.
        index : (N, )
                The index of the nearest neighbor to x in data.

        """
        if isinstance(x, np.ndarray):
            x = mx.array(x)
        elif not isinstance(x, mx.array):
            raise TypeError("`x` must be numpy array or mlx array")

        if x.ndim != 2:
            raise ValueError("`x` must be a 2D array")
        
        if x.shape[1] != self.data.shape[1]:
            raise ValueError("The number of dimensions in `x` and `data` must be the same")

        d, i = _find_nn2(x, self.data)

        # Converting to numpy arrays will evaluate the mx arrays
        return np.array(d, copy=False), np.array(i, copy=False)


@mx.compile
def _find_nn(X, Y):
    """Find the nearest neighbor between a single pooint x and a set of points y.
    
    Parameters 
    ----------
    x :     (N, ) mx.darray
            A single point.
    y :     (M, N) mx.darray
            A set of points.

    Returns
    -------
    index : int
            The index of the nearest neighbor to x in y.
    dist :  float
            The distance between x and its nearest neighbor in y.

    """
    d = mx.sum(mx.subtract(X, Y, stream=mx.gpu) ** 2, axis=1, stream=mx.gpu)
    return mx.argmin(d), mx.min(d).sqrt()

# Vectorized version of the function over the first axis of the first input (i.e. x).
_find_nn_vec = mx.vmap(_find_nn, in_axes=(0, None))

@mx.compile
def _find_nn2(X, Y):
    """A vectorized version of the above.
    
    It seems faster than the vectorized version of the above.
    """
    sx = mx.sum(X**2, axis=1, keepdims=True, stream=mx.gpu)
    sy = mx.sum(Y**2, axis=1, keepdims=True, stream=mx.gpu)
    d = mx.sqrt(-2 * mx.matmul(X, Y.T, stream=mx.gpu) + sx + sy.T, stream=mx.gpu)
    i = mx.argmin(d, axis=1, stream=mx.gpu)
    return d[i], i

