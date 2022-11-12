#include <iostream>
#include "headers/matrix.h"

using namespace std;

//void Matrix::initAndClear()
//{
//    for (size_t i = 0; i < this->rows; i++)
//    {
//        this->matrix[i] = new double[this->cols];
//        for (size_t j = 0; j < this->cols; j++)
//        {
//            this->matrix[i][j] = 0;
//        }
//    }
//}

vector<vector<double>> Matrix::init_matrix(int rows, int cols) {
    vector<vector<double>> vec(rows);
    for (int i = 0; i < rows; i++) {
        vector<double> row(cols);
        for (int j = 0; j < cols; j++) {
            row[j] = 0;
        }
        vec[i] = row;
    }
    return vec;
}

void Matrix::add(double n) {
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            this->matrix[i][j] += n;
        }
    }
}

void Matrix::add(Matrix n) {
    if (rows != n.rows || cols != n.cols) {
        throw invalid_argument("Matrix::add -> Matrices must have the same dimensions");
    }
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            this->matrix[i][j] += n.matrix[i][j];
        }
    }
}

Matrix Matrix::sub(Matrix n) {
    if (rows != n.rows || cols != n.cols) {
        throw invalid_argument("Matrix::sub -> Matrix dimensions must match");
    }
    Matrix result = Matrix(rows, cols);
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            result.matrix[i][j] = this->matrix[i][j] - n.matrix[i][j];
        }
    }
    return result;
}


void Matrix::multiply(double n) {
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            this->matrix[i][j] *= n;
        }
    }
}

// A.K.A. Cross-Product
Matrix Matrix::multiply(Matrix n) {
    if (this->cols != n.rows) {
        throw invalid_argument(
                "Matrix::multiply -> Left Matrix should have the same rows as the columns in the Right Matrix");
    }
    auto *p = new Matrix(this->rows, n.cols);
    for (size_t i = 0; i < p->rows; i++) {
        for (size_t j = 0; j < p->cols; j++) {
            for (size_t k = 0; k < this->cols; k++) {
                //cout << this->matrix[i][k] << " " << n.matrix[k][j] << endl;
                p->matrix[i][j] += this->matrix[i][k] * n.matrix[k][j];
            }
            //cout << endl;
        }
    }
    // why the next 3 lines?
    this->cols = p->cols;
    this->rows = p->rows;
    this->matrix = p->matrix;
    return *p;
}

Matrix Matrix::transpose() {
    auto newMatrix = Matrix(this->cols, this->rows);
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            newMatrix.matrix[j][i] = this->matrix[i][j];
        }
    }
    return newMatrix;
}

void Matrix::printMatrix() const {
    cout << "Matrix with the size: (" << this->rows << "," << this->cols << ")\n";
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            cout << "|" << this->matrix[i][j] << "\t";
        }
        cout << "\n";
    }
    cout << "\n";
}

void Matrix::mapSelf(double (*activation)(double sum)) {
    double expSum = this->map([](double sum) { return exp(sum); }).sum();
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            this->matrix[i][j] = activation(this->matrix[i][j]);
        }
    }
}

Matrix Matrix::map(double (*activation)(double sum)) {
    Matrix result = Matrix(this->rows, this->cols);
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            result.matrix[i][j] = activation(this->matrix[i][j]);
        }
    }
    return result;
}

double Matrix::at(int i, int j) {
    return this->matrix[i][j];
}

void Matrix::set(int i, int j, double num) {
    this->matrix[i][j] = num;
}

Matrix Matrix::hadamard(Matrix n) {
    // Element-wise multiplication
    if (this->rows != n.rows || this->cols != n.cols) {
        throw invalid_argument("Matrix::hadamard -> Matrices must have the same dimensions, Matrix HADAMARD");
    }
    Matrix result = Matrix(this->rows, this->cols);
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            result.matrix[i][j] = this->matrix[i][j] * n.matrix[i][j];
        }
    }
    return result;
}

size_t Matrix::getRows() const {
    return this->rows;
}

size_t Matrix::getCols() const {
    return this->cols;
}

double Matrix::sum() {
    double result = 0;
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            result += this->matrix[i][j];
        }
    }
    return result;
}