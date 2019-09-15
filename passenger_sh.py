import logger
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


DIST_COEF = 1.5
WIDTH = 0.45  # bar width

DATETIMES = [str2time(f'2007-02-20 {hour:02}:00:00') for hour in range(24)]
DISTANCES = list(range(51))


def is_day(date_time):
    return date_time > str2time('2007-02-20 05:00:00') and date_time < str2time('2007-02-20 23:00:00')


def categorize(x, categories):
    idx = (np.abs([(x - dt).seconds for dt in categories])).argmin()
    return idx


def categorize2(x, categories):
    idx = (np.abs([x - dt for dt in categories])).argmin()
    return idx


def main():
    passengers = [0 for _ in DATETIMES]
    stays = [0 for _ in DATETIMES]
    leaves = [0 for _ in DATETIMES]
    day = [0 for _ in DISTANCES]
    night = [0 for _ in DISTANCES]
    vacant = []

    df = pd.read_csv('logs/shanghai/progress_bkup.csv')

    for _, df_ in df.groupby(by='id'):
        # df_ = df_.sort_values(by='time')
        for i in df_.index:
            coo = Coordination(df_.latitude[i], df_.longitude[i])
            if df_.customer[i] > 0 and in_airport(coo):  # pick up at airport
                passengers[categorize(str2time(df_.datetime[i]), DATETIMES)] += 1

        states = []
        for i in range(1, len(df_.index)):
            j = df_.index[i]
            k = df_.index[i - 1]
            coo = Coordination(df_.latitude[k], df_.longitude[k])
            coo_ = Coordination(df_.latitude[j], df_.longitude[j])
            if df_.customer[k] == 0:
                if in_airport(coo) and not in_airport(coo_):
                    vacant.append(coo.dist(coo_) * DIST_COEF)
            elif in_airport(coo):
                # day[categorize()]
                if is_day(str2time(df_.datetime[k])):
                    day[categorize2(coo.dist(coo_) * DIST_COEF, DISTANCES)] += 1
                else:
                    night[categorize2(coo.dist(coo_) * DIST_COEF, DISTANCES)] += 1
                dist_sta = 'short' if near_airport(Coordination(df_.latitude[j], df_.longitude[j])) else 'long'
                states.append({'sta': f'airport->{dist_sta}', 'datetime': df_.datetime[k]})
            elif in_airport(coo_):
                states.append({'sta': '->airport', 'datetime': df_.datetime[k]})
            else:
                states.append({'sta': '->', 'datetime': df_.datetime[j]})

        for i in range(1, len(states)):
            if states[i - 1]['sta'] == '->airport':
                sta = states[i]
                if sta['sta'].startswith('airport->'):
                    stays[categorize(str2time(sta['datetime']), DATETIMES)] += 1
                elif sta['sta'] == '->':
                    leaves[categorize(str2time(sta['datetime']), DATETIMES)] += 1
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

    logger.configure('logs/passengers')
    logger.record_tabular('airport', 'Hongqiao')
    logger.record_tabular('vacant distance sum', np.sum(vacant))
    logger.record_tabular('vacant distance mean', np.mean(vacant))
    logger.record_tabular('vacant distance std', np.std(vacant))
    logger.record_tabular('distance coefficent', DIST_COEF)
    logger.record_tabular('distance sum (day)', np.sum([i for i in range(len(day)) for j in range(day[i])]))
    logger.record_tabular('distance mean (day)', np.mean([i for i in range(len(day)) for j in range(day[i])]))
    logger.record_tabular('distance std (day)', np.std([i for i in range(len(day)) for j in range(day[i])]))
    logger.record_tabular('distance sum (night)', np.sum([i for i in range(len(night)) for j in range(night[i])]))
    logger.record_tabular('distance mean (night)', np.mean([i for i in range(len(night)) for j in range(night[i])]))
    logger.record_tabular('distance std (night)', np.std([i for i in range(len(night)) for j in range(night[i])]))
    logger.dump_tabular()

    plt.bar(np.arange(51)-WIDTH/2, day, width=WIDTH, color='C1', label='day')
    plt.bar(np.arange(51)+WIDTH/2, night, width=WIDTH, color='C0', label='night')
    plt.legend()
    plt.grid(linestyle='--')
    plt.xticks(range(0, 52, 3))
    plt.yticks(range(0, 13, 2))
    plt.ylabel('# trips')
    plt.xlabel('trip distance (km)')
    plt.savefig('logs/shanghai/day_night.jpg')
    plt.show()
    plt.cla()


if __name__ == '__main__':
    main()
