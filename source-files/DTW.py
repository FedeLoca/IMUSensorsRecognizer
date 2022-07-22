import fastdtw
from scipy.spatial import distance
import numpy as np


def dtw(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1, 1), b.reshape(-1, 1))
    cumdist = np.matrix(np.ones((an+1, bn+1)) * np.inf)
    cumdist[0, 0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai, bi] + minimum_cost

    return cumdist[an, bn]


def fast_dtw(a, b):
    return fastdtw.fastdtw(a, b)[0]
