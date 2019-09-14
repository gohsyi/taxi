import numpy as np
from functools import partial


g_passenger = []
g_taxi = []

T = 0
WALKING_TO_LANE = [5, 10]
PACKING_TIME = lambda : np.random.normal(15, 5)


def choose_taxi():
    """
    Returns
    -------
    the id of the taxi
    -1 if the passenger should wait
    """
    return -1


class Lane(object):
    def __init__(self, lam):
        pass

    def step(self):
        pass


class Passenger(object):
    def __init__(self):
        self.taxi = -1
        self.t = T
        self.delta_t = 0

    def step(self):
        if self.taxi < 0:
            self.taxi = choose_taxi()
            if self.taxi > 0:
                self.sta = 'walking'
                self.t = T
                self.delta_t = WALKING_TO_LANE[g_taxi[self.taxi].lane]  # transfer to next state
        elif self.t + self.delta_t == T:
            if self.sta == 'walking':
                # the passenger just arrived at the taxi
                self.sta = 'packing'
                self.t = T
                self.delta_t = PACKING_TIME()
            elif self.sta == 'packing':
                self.sta = 'leaving'



def main():
    global T
    while T < 36000:
        T += 1


if __name__ == '__main__':
    main()
