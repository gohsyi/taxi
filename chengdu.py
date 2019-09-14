import os
import logger
logger.configure('logs/chengdu')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import Coordination
from datetime import datetime
str2time = lambda s: datetime.strptime(s, '%Y/%m/%d %H:%M:%S')

IN = 1
NEAR = 10

SHUANGLIU = Coordination(30.58167, 103.95611)
AIRPORT = SHUANGLIU


class Taxi(object):
    def __init__(self, df):
        date_time = [str2time(time) for time in df.time.values]
        self.id = df.id[df.index[0]]
        self.trajs = [0]
        self.states = []
        self.coo = [Coordination(df.latitude[df.index[0]], df.longitude[df.index[0]])]
        self.customer = [df.customer[df.index[0]]]
        self.datetime = [date_time[0]]
        self.stays = 0
        self.leaves = 0
        self.shorts = 0
        self.longs = 0

        for ii, i in enumerate(df.index[1:]):
            dis = self.coo[-1].dist(Coordination(df.latitude[i], df.longitude[i]))
            dur = (date_time[ii+1] - self.datetime[-1]).seconds / 3600
            # valid data point
            if dis / dur < 200 and dis / dur > 1:
                self.coo.append(Coordination(df.latitude[i], df.longitude[i]))
                self.datetime.append(date_time[ii+1])
                # self.velocity.append(df.velocity[i])
                self.customer.append(df.customer[i])

                if self.customer[-2] != self.customer[-1] or i == len(date_time) - 1:
                    if self.customer[-2] == 0:
                        # self.states.append('')
                        pass
                    elif self.coo[self.trajs[-1]].near(AIRPORT, IN) and self.coo[-1].near(AIRPORT, NEAR):
                        self.states.append('airport->short')
                    elif self.coo[self.trajs[-1]].near(AIRPORT, IN) and not self.coo[-1].near(AIRPORT, NEAR):
                        self.states.append('airport->long')
                    elif not self.coo[self.trajs[-1]].near(AIRPORT, IN) and self.coo[-1].near(AIRPORT, IN):
                        self.states.append('->airport')
                    else:
                        self.states.append('->')
                    self.trajs.append(len(self.customer) - 1)

        for i in range(1, len(self.trajs)):
            coos = self.coo[self.trajs[i - 1]:self.trajs[i]]
            if self.customer[self.trajs[i - 1]] > 0:
                plt.plot([c.longitude for c in coos], [c.latitude for c in coos], alpha=0.6)
            else:
                plt.plot([c.longitude for c in coos], [c.latitude for c in coos], alpha=0.2)

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
        1: 'latitude',
        2: 'longitude',
        3: 'customer',
        4: 'time',
    })


def draw_Shuangliu():
    plt.scatter([SHUANGLIU.longitude], [SHUANGLIU.latitude], c='b', marker='*', label='Shuangliu Airport', zorder=1000)
    c1 = plt.Circle((SHUANGLIU.longitude, SHUANGLIU.latitude), IN/111., color='r', fill=False)
    c2 = plt.Circle((SHUANGLIU.longitude, SHUANGLIU.latitude), NEAR/111., color='g', fill=False)
    plt.gcf().gca().add_artist(c1)
    plt.gcf().gca().add_artist(c2)
    plt.legend()
    plt.grid(linestyle='--')
    plt.show()
    plt.cla()


def main():
    T = []
    for root, dirs, files in os.walk('data/Speed_Prediction'):
        for csv in files:
            if not csv.startswith('.') and csv.endswith('.txt'):
                df = pd.read_csv(os.path.join(root, csv), header=None)
                df = preprocess_df(df)
                for _, df_ in df.groupby('id'):
                    Taxi(df_)

    draw_Shuangliu()

    logger.info('# long distance from Shuangliu:', np.sum([t.longs for t in T]))
    logger.info('# short distance from Shuangliu:', np.sum([t.shorts for t in T]))
    logger.info('# take customer to Shuangliu and stay:', np.sum([t.stays for t in T]))
    logger.info('# take customer to Shuangliu and leave:', np.sum([t.leaves for t in T]))


if __name__ == '__main__':
    main()
    # import matplotlib.pyplot as plt
    # Taxi(preprocess_df(pd.read_csv('data/Taxi_070220/Taxi_10433', header=None)))
    # draw_Hongqiao()
