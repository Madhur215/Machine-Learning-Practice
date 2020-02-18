import numpy as np
import pandas as pd


def run():
    dataset = np.genfromtxt("Data.csv", delimiter=",")
    data = pd.read_csv("Data.csv")
    x = data.iloc[:, -1].values
    y = np.array(data.iloc[:, 1].values)
    print(type(data))
    print(type(dataset))
    print(type(x))
    # res = [[1 for i in range(len(x))] for j in range(1)]


if __name__ == '__main__':
    run()
