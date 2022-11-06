# Polynomial basis linear model data generator
import math
import numpy as np
import GaussianGenerator as gg
import matplotlib.pyplot as plt

# sample times
SAMPLES = 10000


# return the x and y(without adding error) and error
def sample_linear_gen(n, W, std, x=None):
    # start from x^0
    p = 0

    # generate x if x is not provided
    if x is None:
        x = np.random.uniform(low = -1, high = 1)

    inner_product = 0 # the first term

    for i in range(n):
        inner_product += W[i] * math.pow(x, p)
        p += 1

    e, _ = gg.sample_normal_distribution(0, std)

    return x, inner_product, e

def main():
    # size of W == number of basis
    n = int(input('number of basis: '))
    std = math.sqrt(float(input('variance: ')))
    W = [float(x) for x in input('enter W: ').split(' ')]

    data_x = []
    data_y = []
    for _ in range(SAMPLES):
        x, y, e = sample_linear_gen(n, W, std)
        data_x.append(x)
        data_y.append(y + e)

    plt.scatter(data_x, data_y)
    plt.savefig('./plot/linearGenerator')



if __name__ == '__main__':
    main()