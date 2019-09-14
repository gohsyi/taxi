import os
import logger
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

from utils import Coordination
from datetime import datetime
str2time = lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

IN = 1
NEAR = 10

HONGQIAO1 = Coordination(31.19583, 121.34222)
HONGQIAO2 = Coordination(31.19611, 121.3225)
PUDONG = Coordination(31.143333, 121.805278)
AIRPORT = HONGQIAO1, HONGQIAO2


def in_airport(coo):
    return coo.near(AIRPORT[0], IN) or coo.near(AIRPORT[1], IN)


def near_airport(coo):
    return coo.near(AIRPORT[0], IN) or coo.near(AIRPORT[1], NEAR)


class Taxi(object):
    def __init__(self, df):
        date_time = [str2time(time) for time in df.time.values]
        self.id = df.id[0]
        self.trajs = [0]
        self.states = []
        self.coo = [Coordination(df.latitude[0], df.longitude[0])]
        self.datetime = [date_time[0]]
        # self.velocity = [df.velocity[0]]
        self.customer = [df.customer[0]]
        self.stays = 0
        self.leaves = 0
        self.shorts = 0
        self.longs = 0

        for i in range(1, len(date_time)):
            dis = self.coo[-1].dist(Coordination(df.latitude[i], df.longitude[i]))
            dur = (date_time[i] - self.datetime[-1]).seconds / 3600
            # valid data point
            if dis / dur < 200 and dis / dur > 1:
                self.coo.append(Coordination(df.latitude[i], df.longitude[i]))
                self.datetime.append(date_time[i])
                # self.velocity.append(df.velocity[i])
                self.customer.append(df.customer[i])

                if self.customer[-2] != self.customer[-1] or i == len(date_time) - 1:
                    if self.customer[-2] == 0:
                        # self.states.append('')
                        pass
                    elif self.coo[self.trajs[-1]].near(AIRPORT[0], IN) or self.coo[self.trajs[-1]].near(AIRPORT[1], IN):
                        if self.coo[-1].near(AIRPORT[0], NEAR) or self.coo[-1].near(AIRPORT[1], NEAR):
                            self.states.append('airport->short')
                        elif not (self.coo[-1].near(AIRPORT[0], NEAR) or self.coo[-1].near(AIRPORT[1], NEAR)):
                            self.states.append('airport->long')
                    elif self.coo[-1].near(AIRPORT[0], IN) or self.coo[-1].near(AIRPORT[1], IN):
                        self.states.append('->airport')
                    else:
                        self.states.append('->')
                    self.trajs.append(len(self.customer) - 1)

        # for i in range(1, len(self.trajs)):
        #     coos = self.coo[self.trajs[i - 1]:self.trajs[i]]
        #     if self.customer[self.trajs[i - 1]] > 0:
        #         plt.plot([c.longitude for c in coos], [c.latitude for c in coos], alpha=0.6)
        #     else:
        #         plt.plot([c.longitude for c in coos], [c.latitude for c in coos], alpha=0.2)

        for i in range(len(self.trajs)):
            j = self.trajs[i]
            logger.record_tabular('time', self.datetime[j])
            logger.record_tabular('id', self.id)
            logger.record_tabular('latitude', self.coo[j].latitude)
            logger.record_tabular('longitude', self.coo[j].longitude)
            logger.record_tabular('customer', self.customer[j])
            logger.dump_tabular()

        for i in range(1, len(self.states)):
            if self.states[i - 1] == '->airport':
                if self.states[i].startswith('airport->'):
                    self.stays += 1
                elif self.states[i] == '->':
                    self.leaves += 1
            if self.states[i] == 'airport->short':
                self.shorts += 1
            elif self.states[i] == 'airport->long':
                self.longs += 1

        logger.info(self.id, self.states)


def preprocess_df(df):
    return df.rename(columns={
        0: 'id',
        1: 'datetime',
        2: 'longitude',
        3: 'latitude',
        4: 'angle',
        5: 'velocity',
        6: 'customer',
    })


def draw_Hongqiao():
    plt.scatter([HONGQIAO1.longitude], [HONGQIAO1.latitude], c='b', marker='*', label='Hongqiao Airport Terminal 1', zorder=1000)
    c1 = plt.Circle((HONGQIAO1.longitude, HONGQIAO1.latitude), IN/111., color='r', fill=False)
    c2 = plt.Circle((HONGQIAO1.longitude, HONGQIAO1.latitude), NEAR/111., color='g', fill=False)
    plt.gcf().gca().add_artist(c1)
    plt.gcf().gca().add_artist(c2)

    plt.scatter([HONGQIAO2.longitude], [HONGQIAO2.latitude], c='b', marker='*', label='Hongqiao Airport Terminal 2', zorder=1000)
    c1 = plt.Circle((HONGQIAO2.longitude, HONGQIAO2.latitude), IN/111., color='r', fill=False)
    c2 = plt.Circle((HONGQIAO2.longitude, HONGQIAO2.latitude), NEAR/111., color='g', fill=False)
    plt.gcf().gca().add_artist(c1)
    plt.gcf().gca().add_artist(c2)

    plt.legend()
    plt.grid(linestyle='--')
    try:
        plt.show()
    except:
        plt.savefig('logs/shanghai/hongqiao.jpg')
    plt.cla()


def main():
    logger.configure('logs/shanghai')
    T = []
    for root, dirs, files in os.walk('data/Taxi_070220'):
        for csv in files[:5000]:
            if not csv.startswith('.'):
                df = pd.read_csv(os.path.join(root, csv), header=None)
                df = preprocess_df(df)
                T.append(Taxi(df))

    # draw_Hongqiao()

    logger.info('# long distance from Hongqiao:', np.sum([t.longs for t in T]))
    logger.info('# short distance from Hongqiao:', np.sum([t.shorts for t in T]))
    logger.info('# take customer to Hongqiao and stay:', np.sum([t.stays for t in T]))
    logger.info('# take customer to Hongqiao and leave:', np.sum([t.leaves for t in T]))


if __name__ == '__main__':
    main()
    # import matplotlib.pyplot as plt
    # Taxi(preprocess_df(pd.read_csv('data/Taxi_070220/Taxi_10433', header=None)))
    # draw_Hongqiao()
