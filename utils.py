import numpy as np



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
