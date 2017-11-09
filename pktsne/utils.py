import numpy as np

import itertools as IT
import random
from functools import partial, update_wrapper, reduce
import time
from contextlib import contextmanager
import multiprocessing as mp

try:
    from tensorflow import set_random_seed
except ImportError:
    set_random_seed = None


def np_to_sharedarray(size, dtype):
    typecode = dtype.char
    count = reduce(int.__mul__, size, 1)
    mp_arr = mp.Array(typecode, count)
    arr = np.frombuffer(mp_arr.get_obj(), dtype=dtype, count=count)
    return arr.reshape(size)


def sharedarray_to_np(array, dtype):
    return np.frombuffer(array.obj(), dtype)


@contextmanager
def Timer(name):
    print("Starting:", name)
    start = time.time()
    yield
    dt = time.time() - start
    print("Time elapsed: {}: {:0.4f}s".format(name, dt))


def iter_double(iter_):
    for item in iter_:
        yield item
        yield item


def fix_random_seed(seed=1):
    print("Setting numpy random seed")
    np.random.seed(seed)
    if set_random_seed is not None:
        print("Setting tensorflow random seed")
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


def inf_chunk(X, chunk_size):
    yield from IT.cycle(chunk(X, chunk_size))


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
