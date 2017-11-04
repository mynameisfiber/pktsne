import numpy as np

import random
from functools import partial, update_wrapper
import numpy as np

try:
    from tensorflow import set_random_seed
except ImportError:
    set_random_seed = None


def fix_random_seed(seed=42):
    np.random.seed(seed)
    if set_random_seed is not None:
        set_random_seed(seed)


def multiple_gaussians(n_samples, n_dim, n_gaussians):
    g_params = []
    for i in range(n_gaussians):
        mean = np.random.rand(n_dim)
        B = (np.random.rand(n_dim, n_dim) - 0.5) / 10
        cov = np.dot(B, B.T)
        g_params.append((i, (mean, cov)))
    X = np.empty((n_samples, n_dim))
    y = np.empty((n_samples,))
    for i in range(n_samples):
        j, params = random.choice(g_params)
        X[i] = np.random.multivariate_normal(*params)
        y[i] = j
    return X, y


def chunk(X, chunk_size):
    """
    Chunks up an slicable container for when you want to deal with batches.
    >>> map(batch_process, chunk(data, 32))
    """
    for i in range(0, len(X)//chunk_size*chunk_size, chunk_size):
        yield X[i:i+chunk_size]


def wrapped_partial(func, *args, **kwargs):
    """
    Fix because keras looks at loss function's __name__ attribute. From:
    http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func
