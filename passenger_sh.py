import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

from shanghai import str2time, in_airport, near_airport
from utils import Coordination

try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


WIDTH = 0.45  # bar width

DATETIMES = [str2time(f'2007-02-20 {hour:02}:00:00') for hour in range(24)]


def categorize(date_time):
    idx = (np.abs([(str2time(date_time) - dt).seconds for dt in DATETIMES])).argmin()
    return idx


def main():
    passengers = [0 for _ in DATETIMES]
    stays = [0 for _ in DATETIMES]
    leaves = [0 for _ in DATETIMES]

    df = pd.read_csv('logs/shanghai/progress.csv')

    for _, df_ in df.groupby(by='id'):
        # df_ = df_.sort_values(by='time')
        for i in df_.index:
            coo = Coordination(df_.latitude[i], df_.longitude[i])
            if df_.customer[i] > 0 and in_airport(coo):  # pick up at airport
                passengers[categorize(df_.datetime[i])] += 1

        states = []
        for i in range(1, len(df_.index)):
            j = df_.index[i]
            k = df_.index[i - 1]
            if df_.customer[k] == 0:
                # self.states.append('')
                pass
            elif in_airport(Coordination(df_.latitude[k], df_.longitude[k])):
                if near_airport(Coordination(df_.latitude[j], df_.longitude[j])):
                    states.append({'sta': 'airport->short', 'datetime': df_.datetime[k]})
                else:
                    states.append({'sta': 'airport->long', 'datetime': df_.datetime[k]})
            elif in_airport(Coordination(df_.latitude[j], df_.longitude[j])):
                states.append({'sta': '->airport', 'datetime': df_.datetime[k]})
            else:
                states.append({'sta': '->', 'datetime': df_.datetime[j]})

        for i in range(1, len(states)):
            if states[i - 1]['sta'] == '->airport':
                sta = states[i]
                if sta['sta'].startswith('airport->'):
                    stays[categorize(sta['datetime'])] += 1
                elif sta['sta'] == '->':
                    leaves[categorize(sta['datetime'])] += 1
            # if states[i] == 'airport->short':
            #     shorts += 1
            # elif states[i] == 'airport->long':
            #     longs += 1

    plt.bar(range(24), passengers, width=WIDTH, label='passenger flow')
    plt.legend()
    plt.grid(linestyle='--')
    plt.xticks(range(0, 25, 4), [f'{h:02}:00' for h in range(0, 25, 4)])
    plt.yticks(range(0, 30, 4))
    plt.savefig('logs/shanghai/passenger_flow.jpg')
    plt.show()
    plt.cla()

    plt.bar(np.arange(24) - WIDTH/2, leaves, width=WIDTH, label='# leave')
    plt.bar(np.arange(24) + WIDTH/2, stays, width=WIDTH, label='# stay')
    plt.legend()
    plt.grid(linestyle='--')
    plt.xticks(range(0, 25, 4), [f'{h:02}:00' for h in range(0, 25, 4)])
    plt.yticks(range(0, 20, 2))
    plt.savefig('logs/shanghai/leave_stay.jpg')
    plt.show()
    plt.cla()

    stays_to_leaves = gaussian_filter1d((np.array(stays)+1) / (np.array(leaves)+1), sigma=2)
    plt.plot(stays_to_leaves, label='# leave / # stay')
    plt.legend()
    plt.grid(linestyle='--')
    plt.xticks(range(0, 25, 4), [f'{h:02}:00' for h in range(0, 25, 4)])
    plt.savefig('logs/shanghai/leave_to_stay.jpg')
    plt.show()
    plt.cla()


if __name__ == '__main__':
    main()
