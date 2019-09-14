import numpy as np
import pandas as pd

from datetime import datetime
from shanghai import str2time, in_airport
from utils import Coordination

import matplotlib.pyplot as plt


DATETIMES = [str2time(f'2007-02-20 {hour:02}:00:00') for hour in range(24)]


def categorize(date_time):
    idx = (np.abs([(date_time - dt).seconds for dt in DATETIMES])).argmin()
    return idx


def main():
    passengers = [0 for _ in DATETIMES]
    df = pd.read_csv('logs/shanghai/progress.csv')
    df.rename(columns={0: 'latitude', 1: 'longitude', 2: 'id', 3: 'datetime', 4: 'customer'}, inplace=True)

    for _, df_ in df.groupby(by='id'):
        for i in df_.index:
            coo = Coordination(df_.latitude[i], df_.longitude[i])
            if df_.customer[i] > 0 and in_airport(coo):  # pick up at airport
                passengers[categorize(df_.datetime)] += 1

    plt.bar(range(24), passengers)
    plt.show()


if __name__ == '__main__':
    main()
