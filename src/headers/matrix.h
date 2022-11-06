#ifndef PV021_NEURAL_NETWORK_MATH_H
#define PV021_NEURAL_NETWORK_MATH_H

#include <iostream>
#include "activation.h"

#include <vector>

using namespace std;


class Matrix {
private:
    size_t cols;
public:
    size_t getCols() const;

private:
    size_t rows;
    //double **matrix;
    std::vector<std::vector<double>> matrix;

public:
    Matrix(size_t rows, size_t cols) {
        this->cols = cols;
        this->rows = rows;
        if (rows == 0 && cols == 0) {
            return;
        }
        this->matrix = init_matrix(rows, cols);
        //this->initAndClear();
    }

    ~Matrix() {
        // free(this->matrix);
    }

    vector<vector<double>> init_matrix(int rows, int cols);

    double at(int i, int j);

    void set(int i, int j, double num);

    //void initAndClear();
    void add(double n);

    Matrix transpose();

    void add(Matrix n);

    Matrix sub(Matrix n);

    void multiply(double n);

    Matrix multiply(Matrix n);

    Matrix hadamard(Matrix n);

    void mapSelf(double (*activation)(double sum));

    Matrix map(double (*activation)(double sum));

    void printMatrix() const;

    unsigned long long int getRows() const;

    // DEPRECATED
    void randomise();
};

#endif
