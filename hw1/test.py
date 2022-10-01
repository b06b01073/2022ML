from math_util import Coordinate
from math_util import Matrix
import main

A = Matrix([[8, -6, 2], [-6, 7, -4], [2, -4, 3]])

def test_equality():
    assert Matrix([[1, 2], [2, 1]]) == Matrix([[1, 2], [2, 1]]), "Should be equal"
    assert Matrix([[1, 2], [2, 0]]) != Matrix([[1, 2], [2, 1]]), "Should not be equal"

def test_multiplication():
    I = Matrix(row=2, col=2, identity=True)
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [-7, 8]])
    C = Matrix([[-9, 22], [-13, 50]])
    D = Matrix([[3, 4, 5], [6, 7, 8]])
    E = Matrix([[15, 18, 21], [33, 40, 47]])
    Zero = Matrix([[0, 0], [0, 0]])

    # A * B = C
    # A * D = E

    assert A.mul(B) == C
    assert A.mul(C) != B
    assert Zero.mul(A) == Zero
    assert I.mul(A) == A
    assert A.mul(D) == E


def test_transpose():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[1, 3], [2, 4]])
    C = Matrix(row=2, col=2, identity=True)
    assert A.transpose() == B
    assert A.transpose() != C

def test_LU_decomposition():
    A_1 = Matrix([[8, -6, 2], [-6, 7, -4], [2, -4, 3]])
    

    L_1, U_1 = A_1.LU_decomposition()
    # L_1 = Matrix([[8, 0, 0], [-6,	2.5, 0], [2, -2.5, 0]])
    # U_1 = Matrix([[1, -0.75, 0.25], [0, 1, -1], [0, 0, 1]])

    A_2 = Matrix([[1, 1, 1], [3, 1, -3], [1, -2, 5]])
    L_2, U_2 = A_2.LU_decomposition()
    # L_2 = Matrix([[1, 0, 0], [3, -2, 0], [1, -3, 13]])
    # U_2 = Matrix([[1, 1, 1], [0, 1, 3], [0, 0, 1]])

    assert A_1 == L_1.mul(U_1)
    assert A_2 == L_2.mul(U_2)

def test_inverse():
    A = Matrix([[2, 7, 1], [3, -2, 0], [1, 5, 3]])
    A_inverse = A.inverse()
    I = Matrix(row=3, col=3, identity=True)

    assert I == A.mul(A_inverse)
    assert I == A_inverse.mul(A)


def test_scalar_mul():
    A = Matrix([[1, 2], [3, 4]])
    B = A.scalar_mul(2)

    assert B == Matrix([[2, 4], [6, 8]])


def test_Initialize_A():
    data_points = []
    main.read_data(data_points)

    A = main.initialize_A(2, [Coordinate(x=1, y=2), Coordinate(x=3, y=4)])
    B = main.initialize_A(3, [Coordinate(x=1, y=-1), Coordinate(x=-1, y=0)])

    res_A = Matrix([[1, 1], [1, 3]])
    res_B = Matrix([[1, 1, 1], [1, -1, 1]])


    assert A == res_A
    assert B == res_B

def test_sub_add():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[1, 2], [3, 4]])

    assert A + B == Matrix([[2, 4], [6, 8]])
    assert A - B == Matrix([[0, 0], [0, 0]])

if __name__ == '__main__':
    test_equality()
    test_multiplication()
    test_transpose()
    test_LU_decomposition()
    test_inverse()
    test_scalar_mul()
    test_Initialize_A()
    test_sub_add()

    print("OK")