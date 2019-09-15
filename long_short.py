import logger
import numpy as np
from collections import deque
from utils import packing_time as T_PACKING

T = 0
N_RUNS = 10

N_TAXIS = 100
LENGTH = 10000  # simulation length
TRAVEL_TIME = lambda x: x * 60
T_LEAVING = 5
THRESHOLD = 10

RANDOM_LANE = lambda : g_lanes[['short', 'long'][np.random.randint(2)]]

PRIORITY = False
DICHOTOMY = False

g_lanes = {}
g_taxis = []

PRICING_DAY = [0, 14, 14, 14, 16.5, 19, 21.5, 24, 26.5, 29, 31.5,
               34, 36.5, 39, 41.5, 44, 47.6, 51.2, 54.8, 58.4, 62,
               65.6, 69.2, 72.8, 76.4, 80, 83.6, 87.2, 90.8, 94.4, 98,
               101.6, 105.2, 108.8, 112.4, 116, 119.6, 123.2, 126.8, 130.4, 134,
               137.6, 141.2, 144.8, 148.4, 152, 155.6, 159.2, 162.8, 166.4, 170]
PRICING_NIGHT = [0, 18, 18, 18, 21.1, 24.2, 27.3, 30.4, 33.5, 36.6, 39.7,
                 42.8, 45.9, 49, 52.1, 55.2, 59.9, 64.6, 69.3, 74, 78.7,
                 83.4, 88.1, 92.8, 97.5, 102.2, 106.9, 111.6, 116.3, 121, 125.7,
                 130.4, 135.1, 139.8, 144.5, 149.2, 153.9, 158.6, 163.3, 168, 172.7,
                 177.4, 182.1, 186.8, 191.5, 196.2, 200.9, 205.6, 210.3, 215, 219.7]


def enter(taxi, lane):
    taxi.lane = lane
    taxi.state = 'queuing'
    taxi.t = T
    lane.taxis.append(taxi)


def leave(taxi, lane):
    if len(taxi.distances) > 0:
        last_type = 'short' if taxi.distances[-1] <= THRESHOLD else 'long'
        taxi.income_ratio[last_type].append(taxi.incomes[-1] / (T - taxi.last_leaving_time))
    taxi.last_leaving_time = T

    taxi.t = T
    taxi.state = 'delivering'
    taxi.delta_t = TRAVEL_TIME(taxi.passenger.distance)

    lane.taxis.remove(taxi)
    taxi.lane = None


class Lane(object):
    def __init__(self):
        self.passengers = deque([])
        self.taxis = []
        self.continuous_priority_enters = 0

    def reset(self):
        self.passengers = deque([])
        self.taxis = []

    def n_taxis(self):
        return len(self.taxis)

    def n_passengers(self):
        return len(self.passengers)

    def step(self):
        if len(self.taxis) == 0:
            return

        # choose the next taxi
        if PRIORITY:
            priority_taxis = [taxi for taxi in self.taxis if len(taxi.distances) > 0 and taxi.distances[-1] <= THRESHOLD]
            if self.continuous_priority_enters < 3 and len(priority_taxis) > 0:
                taxi = priority_taxis[0]
                self.continuous_priority_enters += 1
            else:
                taxi = self.taxis[0]
                self.continuous_priority_enters = 0
        else:
            taxi = self.taxis[0]

        if taxi.state == 'queuing' and len(self.passengers) > 0:
            taxi.state = 'packing'
            taxi.t = T
            taxi.delta_t = T_PACKING()
            taxi.passenger = self.passengers.popleft()


class Passenger(object):
    def __init__(self, distance):
        self.distance = distance


class Taxi(object):
    def __init__(self, id):
        self.id = id
        self.state = None
        self.passenger = None
        self.last_leaving_time = 0
        self.t = 0
        self.delta_t = 0
        self.distances = []
        self.incomes = []
        self.income_ratio = {'short': [], 'long': []}
        self.lane = None

    def step(self):
        if self.state == 'packing' and self.t + self.delta_t == T:
            self.t = T
            self.state = 'leaving'
            self.delta_t = T_LEAVING

        elif self.state == 'leaving' and self.t + self.delta_t == T:
            leave(self, self.lane)

        elif self.state == 'delivering' and self.t + self.delta_t == T:
            self.t = T
            self.state = 'returning'
            self.distances.append(self.passenger.distance)
            self.incomes.append(PRICING_DAY[self.passenger.distance])
            # if T < 8000:
            #     self.income += PRICING_DAY[self.passenger.distance]
            # else:
            #     self.income += PRICING_NIGHT[self.passenger.distance]
            self.passenger = None

        elif self.state == 'returning' and self.t + self.delta_t == T:
            # short_pn = (g_lanes['short'].n_taxis() + 1) / (g_lanes['short'].n_passengers() + 1)
            # long_pn = (g_lanes['long'].n_taxis() + 1) / (g_lanes['long'].n_passengers() + 1)
            # if not PRIORITY:
            #     g_lanes[['short', 'long'][np.random.randint(2)]].taxis.append(self)
            # elif short_pn <= long_pn:
            #     g_lanes['short'].taxis.append(self)
            # else:
            #     g_lanes['long'].taxis.append(self)
            enter(self, RANDOM_LANE())

        assert self.state == 'queuing' or self.t + self.delta_t >= T


def main():
    logger.configure('logs/long_short')

    global T, PRIORITY, DICHOTOMY

    for PRIORITY, DICHOTOMY in [(True, False), (False, True), (False, False)]:
        income_means, income_stds = [], []
        short_ratios, long_ratios = [], []
        short_passengers, long_passengers = [], []
        for seed in range(N_RUNS):
            np.random.seed(seed)
            T = 0
            g_lanes.clear()
            g_lanes.update({'short': Lane(), 'long': Lane()})
            # short_passengers, long_passengers = [], []
            enter_passengers = np.random.poisson(0.1, size=LENGTH)

            g_taxis.clear()
            for i in range(N_TAXIS // 2):
                g_taxis.append(Taxi(i))
                enter(g_taxis[-1], g_lanes['short'])
            for i in range(N_TAXIS // 2):
                g_taxis.append(Taxi(i + N_TAXIS // 2))
                enter(g_taxis[-1], g_lanes['long'])

            while T < LENGTH:
                if enter_passengers[T]:
                    if np.random.rand() < 4/7:
                        lane = g_lanes['long'] if DICHOTOMY else RANDOM_LANE()
                        lane.passengers.append(Passenger(np.random.randint(11, 51)))
                    else:
                        lane = g_lanes['short'] if DICHOTOMY else RANDOM_LANE()
                        lane.passengers.append(Passenger(np.random.randint(1, 11)))

                g_lanes['short'].step()
                g_lanes['long'].step()
                for taxi in g_taxis:
                    taxi.step()

                short_passengers.append(len(g_lanes['short'].passengers))
                long_passengers.append(len(g_lanes['long'].passengers))
                T += 1

            incomes = [np.sum(t.incomes) for t in g_taxis]

            income_means.append(np.mean(incomes))
            income_stds.append(np.std(incomes))
            short_ratios.append(np.mean([r for t in g_taxis for r in t.income_ratio['short']]))
            long_ratios.append(np.mean([r for t in g_taxis for r in t.income_ratio['long']]))

        logger.info(income_means)
        logger.info(income_stds)
        logger.record_tabular('*priority*', PRIORITY)
        logger.record_tabular('*dichotomy*', DICHOTOMY)
        logger.record_tabular('income mean', np.mean(income_means))
        logger.record_tabular('income std', np.mean(income_stds))
        logger.record_tabular('short income ratio mean', np.mean(short_ratios))
        logger.record_tabular('short income ratio std', np.std(short_ratios))
        logger.record_tabular('long income ratio mean', np.mean(long_ratios))
        logger.record_tabular('long income ratio std', np.std(long_ratios))
        logger.record_tabular('# short lane passengers', np.mean(short_passengers))
        logger.record_tabular('# long lane passengers', np.mean(long_passengers))
        logger.dump_tabular()


if __name__ == '__main__':
    main()
