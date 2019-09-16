import numpy as np
import math
import pandas as pd



def packing_time(mean=20, std=5, low=0, high=40):
    pt = np.ceil(np.random.normal(mean, std))
    if pt < low:
        return low
    elif pt > high:
        return high
    else:
        return pt


class Coordination(object):
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def output(self):
        print(f'{self.latitude} N, {self.longitude} E')

    def near(self, coo, threshold=10):
        return self.dist(coo) <= threshold

    def dist(self, coo):
        return np.sqrt(
            (self.latitude - coo.latitude) ** 2 +
            (self.longitude - coo.longitude) ** 2
        ) * 111


def cal_weight(x):
    # x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
    rows = x.index.size
    cols = x.columns.size
    k = 1.0 / math.log(rows)

    x = np.array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = np.array(lnf)

    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                if np.isinf(p):
                    p = 1
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij

    lnf = pd.DataFrame(lnf)
    E = lnf
    d = 1 - E.sum(axis=0)
    w = [[None] * 1 for i in range(cols)]

    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj

    w = pd.DataFrame(w)
    return w
