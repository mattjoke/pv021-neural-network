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
        return;
    }
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            this->matrix[i][j] += n.matrix[i][j];
        }
    }
}

Matrix Matrix::sub(Matrix n) {
    if (rows != n.rows || cols != n.cols) {
        throw "Matrix dimensions must match";
        return {0, 0};
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
    if (this->cols != n.rows && this->rows != n.cols) {
        cout << "Columns of A must match rows of B." << endl;
        cout << "A: " << this->rows << "x" << this->cols << endl;
        cout << "B: " << n.rows << "x" << n.cols << endl;
        throw invalid_argument("Left Matrix should have the same rows as the columns in the Right Matrix");
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


// DEPRECATED
void Matrix::randomise() {
    for (size_t i = 0; i < this->rows; i++) {
        for (size_t j = 0; j < this->cols; j++) {
            this->matrix[i][j] =  1; // std::rand() % 10;
        }
    }
}


double Matrix::at(int i, int j) {
    return this->matrix[i][j];
}

void Matrix::set(int i, int j, double num) {
    this->matrix[i][j] = num;
}

Matrix Matrix::hadamard(Matrix n) {
    return this->multiply(n.transpose());
}

unsigned long long int Matrix::getRows() const {
    return this->rows;
}