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


def chunk(X, chunk_size):
    """
    Chunks up an slicable container for when you want to deal with batches.
    >>> map(batch_process, chunk(data, 32))
    """
    for i in range(0, len(X), chunk_size):
        yield X[i:i+chunk_size]


def wrapped_partial(func, *args, **kwargs):
    """
    Fix because keras looks at loss function's __name__ attribute. From:
    http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func
