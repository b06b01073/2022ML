from copy import deepcopy
from math_util import Coordinate, Matrix
import matplotlib.pyplot as plt

file_path = "testfile.txt"

def read_data(data_points):
    with open(file_path, 'r') as f:
        for line in f.readlines():
            x = float(line.split(',')[0])
            y = float(line.split(',')[1])
            c = Coordinate(x, y)
            data_points.append(c)
    

def plot(n, X, data_points, file_name):
    data_x = [c.x for c in data_points]
    data_y = [c.y for c in data_points]

    predict_x = deepcopy(data_x)
    predict_y = []

    for x in predict_x:
        y = 0
        for i in reversed(range(n)):
            y += X.matrix[i][0] * pow(x, i)
        predict_y.append(y)



    plt.scatter(data_x, data_y) 
    plt.plot(predict_x, predict_y)
    plt.title(file_name)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig(file_name)     
    plt.clf()

def initialize_A(n, data_points):
    row = len(data_points)
    col = n

    # [
    #     x^0(data_points[0].x) x^1(data_points[0].x) x^2(data_points[0].x) ....
    #     x^0(data_points[1].x) x^1(data_points[1].x) x^2(data_points[1].x) ....
    # ]
    return Matrix([[pow(data_points[r].x, c) for c in range(col)] for r in range(row)])

def get_LSE_error(n, X, data_points):
    error = 0
    for i in range(len(data_points)):
        prediction = 0
        for j in range(n):
            prediction += X.matrix[j][0] * pow(data_points[i].x, j)
        error += (prediction - data_points[i].y) ** 2

    return error

def LSE(n, lam, data_points):
    A = initialize_A(n, data_points)
    A_trans = A.transpose()
    I = Matrix(row=A.col, col=A.col, identity=True)
    b = Matrix([[data.y] for data in data_points])
    X = (A_trans.mul(A) + I.scalar_mul(lam)).inverse().mul(A_trans).mul(b)
    

    print("\nLSE:")

    print("Fitting line: ", end='')
    for i in reversed(range(n)):
        print(f'{X.matrix[i][0]}X^{i}' + (' + ' if i != 0 else ''), end='')

    print(f'\nTotal Error: {get_LSE_error(n, X, data_points)}')

    plot(n, X, data_points, "LSE")

    
def Newton(n, data_points, episilon=1e-10, max_iter=1e4):
    A = initialize_A(n, data_points)
    A_trans = A.transpose()

    # initial point
    X = Matrix([[0] for _ in range(n)])
    b = Matrix([[data.y] for data in data_points])
    zero_grad = Matrix(row=A_trans.row, col=b.col)
    gradient = A_trans.mul(A).scalar_mul(2).mul(X) - A_trans.mul(b).scalar_mul(2)
    
    # assume that hessian always exist 
    hessian = A_trans.mul(A).scalar_mul(2)

    # 每次loop都要計算gradient過於耗時，設定一個max_iter，超過max_iter直接中止 
    iter = 0

    while True:
        X_new = X - hessian.inverse().mul(gradient)
        max_diff = float('-inf')
        for i in range(X.row):
            for j in range(X.col):
                max_diff = max(abs(X_new.matrix[i][j] - X.matrix[i][j]), max_diff)


        X = X_new
        gradient = A_trans.mul(A).scalar_mul(2).mul(X) - A_trans.mul(b).scalar_mul(2)
        hessian = A_trans.mul(A).scalar_mul(2)

        if max_diff < episilon or gradient == zero_grad or iter > max_iter:
            break

        
        iter += 1

    print("\nNewton's Method:")

    print("Fitting line: ", end='')
    for i in reversed(range(n)):
        print(f'{X.matrix[i][0]}X^{i}' + (' + ' if i != 0 else ''), end='')

    print(f'\nTotal Error: {get_LSE_error(n, X, data_points)}')

    plot(n, X, data_points, "Newton's Method")


def main():
    n = int(input('Enter the number of bases: '))
    lam = float(input('Enter Lambda: '))
    
    data_points = []
    read_data(data_points)
    
    LSE(n, lam, data_points)
    Newton(n, data_points)
    



if __name__ == '__main__':
    main()