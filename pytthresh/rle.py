"""
Run-length Encoding (RLE)
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def encode(x):
    """
    Assumes x is binary.

    The first run is assumed to consist of 1s (it may have length 0)
    """

    result = np.zeros(len(x), dtype=np.uint32)
    where = 0
    last = 1
    counter = 0
    for i in x:
        if last == i:
            counter += 1
        else:
            result[where] = counter
            where += 1
            last = i
            counter = 1
    result[where] = counter
    return result[: where + 1]


@jit(nopython=True, cache=True)
def decode(v):
    result = np.zeros(np.sum(v), dtype=np.uint64)
    where = 0
    current = 1
    for i in range(len(v)):
        result[where : where + v[i]] = current
        where += v[i]
        current = 1 - current
    return result
