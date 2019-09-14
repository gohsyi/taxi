import os


def main():
    for root, dirs, files in os.walk('data/Taxi_070220'):
        for csv in files:
            if not csv.startswith('.'):
                path = os.path.join(root, csv)
                customer = open(path).readline().strip()[-1]
                if customer == '0' or customer == '1':
                    continue
                print('removing', path)
                os.remove(path)

if __name__ == '__main__':
    main()
