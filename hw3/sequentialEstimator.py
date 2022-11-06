import math
import numpy as np
import matplotlib.pyplot as plt
import GaussianGenerator as gg

# the estimation is considered to be converge if the difference of variance and mean between iterations is less than episilon
epsilon = 1e-8 
max_iter = int(1e7)


# implement Welford's method
def main():


    dist_mean = float(input('mean: '))
    dist_var = float(input('variance: '))
    dist_std = math.sqrt(dist_var)


    with open('problem2.txt', 'w') as f:
        f.write(f'Data point source function: N({dist_mean}, {dist_var})\n')
        size = 0
        mean = 0
        diff_square = 0
        data_x = []

        for _ in range(max_iter):
            data, _ = gg.sample_normal_distribution(dist_mean, dist_std)

            data_x.append(data)
            
            f.write(f'Add data point: {data}\n')
            # incremental mean update
            size += 1
            new_mean = mean + (data - mean) / size
            new_diff_square = diff_square + (data - new_mean) * (data - mean)

            var = new_diff_square / (size - 1) if size > 1 else 0

            f.write(f'Mean = {new_mean}\tVariance = {var}\n')


            # converge
            if max(abs(new_mean - mean), abs(new_diff_square - diff_square)) <= epsilon:
                break

            mean = new_mean
            diff_square = new_diff_square

        plt.hist(data_x, bins=60)
        plt.savefig('./plot/sampleDist')


if __name__ == '__main__':
    main()