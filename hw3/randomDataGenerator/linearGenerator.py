# Polynomial basis linear model data generator
import math
import numpy as np
import GaussianGenerator as gg

# sample times
SAMPLES = 1

def main():
    # size of W == number of basis
    n = int(input('number of basis: '))
    std = math.sqrt(float(input('variance: ')))
    W = [float(x) for x in input('enter W: ').split(' ')]

    for _ in range(SAMPLES):
        
        # start from x^0
        p = 0
        x = np.random.uniform(low = -1, high = 1)
        inner_product = 0 # the first term

        for i in range(n):
            inner_product += W[i] * math.pow(x, p)
            p += 1

            

        e, _ = gg.sample_normal_distribution(0, std)

        y = e + inner_product
        print(x, y)



if __name__ == '__main__':
    main()