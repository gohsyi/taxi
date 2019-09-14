import numpy as np


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
