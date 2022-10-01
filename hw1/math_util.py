from copy import deepcopy

class Coordinate():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Matrix():
    def __init__(self, matrix=None, row=None, col=None, identity=False, episilon=1e-8):

        # Initialize a 0 matrix if matrix is not given

        self.matrix = [[0 for _ in range(row)] for _ in range(col)] if matrix is None else matrix
        self.row = len(self.matrix)
        self.col = len(self.matrix[0])

        # 處理浮點數誤差
        self.episilon = episilon

        if identity:
            if self.row != self.col:
                raise(ValueError('The matrix is not square'))
            self.matrix = [[1 if i == j else 0 for i in range(self.row)] for j in range(self.col)]

    def transpose(self):
        return Matrix([[self.matrix[r][c] for r in range(self.row)] for c in range(self.col)])

    # Matrix addition
    def __add__(self, B):
        assert isinstance(B, Matrix)
        assert self.col == B.col and self.row == B.row

        res = Matrix(row=self.row, col=self.col)
        for i in range(self.row):
            for j in range(self.col):
                res.matrix[i][j] = self.matrix[i][j] + B.matrix[i][j]

        return res

    # matrix multiplication
    def mul(self, B):
        if not isinstance(B, Matrix):
            raise(TypeError('The given argument is not a Matrix type.'))
        if self.col != B.row:
            raise(ValueError('The size of matrices does not match.'))

        new_row = self.row
        new_col = B.col
        res_mat = [[0 for _ in range(new_col)] for _ in range(new_row)]

        for i in range(new_row):
            for j in range(new_col):
                for k in range(self.col):
                    res_mat[i][j] += self.matrix[i][k] * B.matrix[k][j]
        
        return Matrix(res_mat)


    def scalar_mul(self, coef):
        res = deepcopy(self)
        for i in range(self.row):
            for j in range(self.col):
                res.matrix[i][j] *= coef
        return res

        
    def inverse(self):
        # assume the matrix is invertible and LU decomposible at this point
        # steps
        # 1. A = LU
        # 2. calculate L^-1 and U^-1
        # 3. A^-1 = U^-1 * L^-1
        L, U = self.LU_decomposition()
        L_inverse = self.__get_L_inverse(deepcopy(L))
        U_inverse = self.__get_U_inverse(deepcopy(U))

        

        return U_inverse.mul(L_inverse)


    def __get_U_inverse(self, U):
        row = U.row
        col = U.col
        res = Matrix(row=row, col=col, identity=True)

        for i in reversed(range(row)):
            coef = 1 / U.matrix[i][i]
            
            # 把整個row根據leading term調整
            for j in range(col):
                res.matrix[i][j] *= coef
                U.matrix[i][j] *= coef


            # 消去
            for k in range(i):
                coef = U.matrix[k][i] / U.matrix[i][i]
                for j in range(col):
                    res.matrix[k][j] -= res.matrix[i][j] * coef
                    U.matrix[k][j] -= U.matrix[i][j] * coef
            
        return res
            
    def __get_L_inverse(self, L):
        row = L.row
        col = L.col
        res = Matrix(row = row, col = col, identity=True)
        for i in range(row):
            coef = 1 / L.matrix[i][i]

            for j in range(col):
                res.matrix[i][j] *= coef
                L.matrix[i][j] *= coef

            for k in range(i + 1, row):
                coef = L.matrix[k][i] / L.matrix[i][i]
                for j in range(col):
                    res.matrix[k][j] -= res.matrix[i][j] * coef
                    L.matrix[k][j] -= L.matrix[i][j] * coef

        return res

    def LU_decomposition(self):
        # using the Crout's method
        if self.col != self.row:
            raise(ValueError('The matrix is not square.'))

        n = self.row

        L = Matrix([[0 for _ in range(n)] for _ in range(n)])
        U = Matrix(row=n, col=n, identity=True)


        for i in range(n):
            for j in range(n):
                if j <= i:
                    L.matrix[i][j] = self.matrix[i][j]

                    # i以後U.matrix[i][k]都是0
                    for k in range(j):
                        L.matrix[i][j] -= L.matrix[i][k] * U.matrix[k][j]
                else:
                    U.matrix[i][j] = self.matrix[i][j]

                    # j以後L.matrix[i][k]都是0，這邊的else block j > i
                    for k in range(i):
                        U.matrix[i][j] -= L.matrix[i][k] * U.matrix[k][j]
                    U.matrix[i][j] /= L.matrix[i][i]

        return L, U

    # for testing
    def __str__(self):
        s = 'Matrix([\n'
        for i in range(self.row):
            for j in range(self.col):
                s += str(self.matrix[i][j]) + ' '
            s += '\n'
        s += '])'

        return s

    # for testing
    def __eq__(self, M):
        if not isinstance(M, Matrix):
            return False
        if M.row != self.row or M.col != self.col:
            return False


        for i in range(self.row):
            for j in range(self.col):
                if abs(M.matrix[i][j] - self.matrix[i][j]) > self.episilon:
                    return False
        return True

    def __ne__(self, M):
        return not self == M
