# Baysian Linear regression
import linearGenerator as lg
import math 

import matplotlib.pyplot as plt
import numpy as np

episilon = 1e-4
fig, ax = plt.subplots(2,2)

def plot_ground_truth(n, W, var):



    # x is in [-2, 2] in order to show the same result as spec 
    ground_truth_x = np.linspace(start = -2, stop = 2)
    ground_truth_y = []

    pos_var_x = np.linspace(start = -2, stop = 2)
    pos_var_y = []

    neg_var_x = np.linspace(start = -2, stop = 2)
    neg_var_y = []


    std = math.sqrt(var)
    for x in ground_truth_x:
        y = lg.sample_linear_gen(n, W, std, x)[1]
        ground_truth_y.append(y)
        pos_var_y.append(y + var)
        neg_var_y.append(y - var)

    ax[0][0].set_xlim(-2, 2)
    ax[0][0].set_ylim(-20, 20)
    ax[0][0].plot(ground_truth_x, ground_truth_y, color='black')
    ax[0][0].plot(pos_var_x, pos_var_y, color='red')
    ax[0][0].plot(neg_var_x, neg_var_y, color='red')
    ax[0][0].title.set_text('Ground Truth')

    
def plot_result(file_name, data_x, data_y, mean, n, a, b, design_matrix, cov_matrix, step=-1):
    # here mean is w 
    X = np.linspace(start = -2, stop = 2)
    predictions = []
    pos_var_y = []
    neg_var_y = []
    var = 0

    var_matrix = np.linalg.inv(cov_matrix)

    for x in X:
        prediction = 0
        for i in range(n):
            prediction += mean[i] * pow(x, i)
        predictions.append(prediction)

    for i in range(len(X)):
        x = X[i]
        design_matrix = np.array([pow(x, j) for j in range(n)])
        var = a + design_matrix.dot(var_matrix).dot(design_matrix.T).item()
        pos_var_y.append(predictions[i] + var)
        neg_var_y.append(predictions[i] - var)
    

    

    grid_pos = (0, 1)
    plot_title = 'predict result'
    if step == 10:
        grid_pos= (1, 0)
        plot_title = '10 incomes'
    elif step == 50:
        grid_pos = (1, 1)
        plot_title = '50 incomes'
    ax[grid_pos[0]][grid_pos[1]].set_xlim(-2, 2)
    ax[grid_pos[0]][grid_pos[1]].set_ylim(-20, 20)
    
    ax[grid_pos[0]][grid_pos[1]].plot(X, predictions, color='black')
    ax[grid_pos[0]][grid_pos[1]].plot(X, pos_var_y, color='red')
    ax[grid_pos[0]][grid_pos[1]].plot(X, neg_var_y, color='red')
    ax[grid_pos[0]][grid_pos[1]].scatter(data_x, data_y)
    ax[grid_pos[0]][grid_pos[1]].title.set_text(plot_title)
    

def write_result(f, n, x, noise_y, mean, cov, predict_mean, predict_var):
    f.write(f'Add data point ({x}, {noise_y})\n')
    f.write('Posterior mean:\n')
    for i in range(n):
        f.write(f'\t{mean[i]}\n')
    f.write('\nPosterior variance:\n')
    for i in range(n):
        for j in range(n):
            # note that the output is variance
            f.write(f'{1 / cov[i][j]}')
            if j != n - 1:
                f.write(', ')
        f.write('\n')

    f.write(f'Predictive distribution ~ N({predict_mean}, {predict_var})')
    f.write('\n\n')

def main():
    b = float(input('b(precision): '))
    n = int(input('n(number of basis): '))

    # the notation here is different from the note
    a = float(input('a(variance): '))
    W = [float(x) for x in input('w(weights): ').split(' ')]

    std = math.sqrt(a)

    plot_ground_truth(n, W, a)

    with open('problem3.txt', 'w') as f:
        data_x = []
        data_y = []

        precision = 1 / a
        precision_matrix = b * np.identity(n)

        old_mean = [0 for _ in range(n)]
        old_cov_matrix = [[0 for _ in range(n)] for _ in range(n)]
        design_matrix = None
        cov_matrix = None

        step = 0
        while True:
            x, y, e = lg.sample_linear_gen(n, W, std)
            noise_y = y + e
            data_x.append(x)
            data_y.append(noise_y)


            design_matrix = np.array([[pow(x, i) for i in range(n)] for x in data_x])

            
            Y = np.array([y for y in data_y]).T

            # (n, n)
            cov_matrix = precision * np.dot(design_matrix.T, design_matrix) + precision_matrix

            # (n, 1)
            new_mean = np.dot(np.linalg.inv(cov_matrix), (precision * np.dot(design_matrix.T, Y) + precision_matrix.dot(old_mean)))

            predict_mean = design_matrix[-1].dot(new_mean).item()

            # note again the "a" here is not the same "a" in the note
            predict_var = a + design_matrix[-1].dot(np.linalg.inv(cov_matrix)).dot(design_matrix.T)[-1].item()


            write_result(f, n, x, noise_y, new_mean, cov_matrix, predict_mean, predict_var)

            step += 1
            if step == 10 or step == 50:
                plot_result(f'{step}Incomes', data_x, data_y, new_mean, n, a, b, design_matrix, cov_matrix, step)


            # check convergence
            max_mean_diff = float('-inf')
            for i in range(n):
                max_mean_diff = max(max_mean_diff, abs(new_mean[i] - old_mean[i]))

            if max_mean_diff <= episilon and step >= 50:
                break

            old_mean = new_mean
        
        plot_result(f'Predict Result', data_x, data_y, new_mean, n, a, b, design_matrix, cov_matrix)


            



if __name__ == '__main__':
    main()
    plt.tight_layout()
    plt.savefig('./plot/BaysianLinearRegression')