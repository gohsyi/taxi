import numpy as np
import pandas as pd
import logger
import seaborn as sns
import matplotlib.pyplot as plt
from utils import packing_time as T_PACKING

T = 0
M, N = 40, 20
N_RUNS = 30

T_TURN, T_STRAIGHT = 5, 5

n_taxis, n_bills = 0, 0
occupied = []


class Lane(object):
    def __init__(self, id, n_states, lam=0.05, steps=10000):
        self.id = id
        self.n_states = n_states
        self.enter = np.random.poisson(lam, size=steps)
        self.n_passengers = 0
        self.t = 0
        self.delta_t = 0
        self.state = -1

    def step(self):
        global n_taxis, n_bills
        if self.enter[T]:
            self.n_passengers += 1

        if self.state < 0 and self.n_passengers > 0 and n_taxis > 0:
            self.n_passengers -= 1
            n_taxis -= 1
            self.next_state()

        if self.state == self.id and occupied[self.id]:
            self.delta_t += 1

        if self.state >= 0 and self.t + self.delta_t == T:
            if self.state == self.id:
                occupied[self.id + 1] = True
            elif self.state == self.id + 2:
                occupied[self.id + 1] = False
            self.next_state()

        if self.state > self.n_states:
            self.state = -1
            n_bills += 1

    def next_state(self):
        self.state += 1
        self.t = T
        if self.id == self.state or self.id == self.state + 2:
            self.delta_t = T_TURN
        elif self.id == self.state + 1:
            self.delta_t = T_PACKING()
        else:
            self.delta_t = T_STRAIGHT


def main():
    logger.configure('logs/simulate')
    global T, n_bills, n_taxis, occupied
    results = []
    for n_lanes in range(2, 10):
        bills, n_taxis_left, n_passengers_left = [], [], []
        for seed in range(N_RUNS):
            np.random.seed(seed)
            occupied = [False for _ in range(n_lanes + 1)]
            T, n_bills, n_taxis, sta = 0, 0, 0, 0
            lanes = [Lane(i, n_lanes+1, lam=0.1/n_lanes) for i in range(n_lanes)]
            enter = np.random.poisson(0.1, size=10000)
            while T < 10000:
                if sta == 0:
                    if n_taxis < M:
                        n_taxis += enter[T]
                    else:
                        sta = 1
                elif n_taxis < N:
                    sta = 0
                for lane in lanes:
                    lane.step()
                T += 1
            bills.append(n_bills)
            n_taxis_left.append(n_taxis)
            n_passengers_left.append(np.sum([lane.n_passengers for lane in lanes]))

        results.append(bills)

        logger.record_tabular('lanes', n_lanes)
        logger.record_tabular('bills mean', np.mean(bills))
        logger.record_tabular('bills std', np.std(bills))
        logger.record_tabular('taxis mean', np.mean(n_taxis_left))
        logger.record_tabular('passengers mean', np.mean(n_passengers_left))
        logger.dump_tabular()

    df = pd.DataFrame(np.reshape(results, -1)).rename(columns={0: '# bills'})
    df.insert(0, '# lanes', [i for i in range(2, 10) for _ in range(N_RUNS)], True)
    sns.boxplot(x='# lanes', y='# bills', data=df, showmeans=True, meanline=True)
    plt.grid(linestyle='--')
    plt.savefig('logs/simulate/boxplot.jpg')
    plt.show()


if __name__ == '__main__':
    main()
