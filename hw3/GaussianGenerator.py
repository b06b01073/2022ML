# Univariate gaussian data generator
import math
import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 100000
plt_path = './plot/GaussianGenerator'

def sample_normal_distribution(mean, std):
    return __Box_Muller(mean, std)

# Implement the Box_Muller algorithm
def __Box_Muller(mean, std):
    u = np.random.uniform(low = 0, high = 1)
    v = np.random.uniform(low = 0, high = 1)


    X = mean + std * math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v) 
    Y = mean + std * math.sqrt(-2 * math.log(u)) * math.sin(2 * math.pi * v)

    return X, Y
    


def main():
    mean = float(input('mean: '))
    var = float(input('variance: '))

    std = math.sqrt(var)

    data_x = []

    for _ in range(SAMPLES):
        x, _ = sample_normal_distribution(mean, std)
        data_x.append(x)

    # plt.scatter(data_x, data_y)
    plt.hist(data_x, bins=60)
    plt.savefig(plt_path)






if __name__ == '__main__':
    main()