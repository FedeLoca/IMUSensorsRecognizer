import math

import numpy as np
from math import sqrt, exp
from scipy.stats import skew
from scipy.signal import correlate
from scipy.signal import welch


def min_max_diff(ls):
    if ls.size == 0:
        return 0
    return max(ls) - min(ls)


def rms(ls):
    if not ls:
        return 0
    return sqrt(sum(n * n for n in ls) / len(ls))


def symmetry(ls):
    if ls.size == 0:
        return 0
    return skew(ls)


def zero_crossing(ls):
    m = min_max_diff(ls) / 2
    return sum(
        [
            (ls[x] < m < ls[x + 1])
            or (ls[x] > m > ls[x + 1])
            for x in range(len(ls) - 1)
        ]
    )


def correlation(ls1, ls2):
    if len(ls1) > 1 and len(ls2) > 1:
        if (np.std(ls1) * np.std(ls2)) == 0:
            return 0
        return np.cov(ls1, ls2)[0][1] / (np.std(ls1) * np.std(ls2))
    return 0


def crossed_correlation(ls1, ls2):
    if not ls1 or not ls2:
        return 0
    return max(correlate(ls1, ls2))


def peaks_number(ls):
    # peaks, _ = spsignal.find_peaks(array1)
    # peaks.size
    if ls.size == 0:
        return 0
    return len(list(filter(lambda x: x is True, [ls[i - 1] > ls[i] < ls[i + 1] or ls[i - 1] < ls[i] > ls[i + 1] for i in
                                                 range(1, len(ls) - 1)])))


def energy(ls):
    result = sum(welch(ls)[1])
    if math.isnan(result) or math.isinf(result):
        pass
        # print(ls)
        # print(np.count_nonzero(np.isnan(ls)))
    return result


def magnitude(ls1, ls2, ls3):
    if ls1.size == 0 or ls2.size == 0 or ls3.size == 0:
        return 0
    s = [sqrt(x**2 + y**2 + z**2) for (x, y, z) in zip(ls1, ls2, ls3)]
    s = sum(s) / len(s)
    return s


def cost_function(x, curr, t_sens):
    return exp(- x * (abs(curr - t_sens) / 60000))


def abs_cost_function(x, curr, t_sens):
    return exp(- x * (abs(curr - t_sens) / 60000))


def feature(curr, old, stat, x):
    def calculate_feature(ls):
        status = stat
        res = 0
        for e in ls:
            if status:
                res = old
            else:
                res += cost_function(x, curr, e.time)
            status = e.status == "ON"
        return res, status
    return calculate_feature


def feature_on_off(curr, x, value):
    def calculate_feature(ls):
        res = 0
        for e in ls:
            if e.status == value:
                res += cost_function(x, curr, e.time)
        return res
    return calculate_feature


def temporal_feature(curr, x):
    def calculate_feature(ls):
        res = 0
        for e in ls:
            res += cost_function(x, curr, e[1]) * e[0]
        return res
    return calculate_feature
