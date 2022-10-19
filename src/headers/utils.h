#ifndef PV021_NEURAL_NETWORK_MATH_H
#define PV021_NEURAL_NETWORK_MATH_H

#include <iostream>
#include "activation.h"

#include <vector>

using namespace std;

class Matrix
{
private:
    size_t cols;
    size_t rows;
    double **matrix;
    std::vector<std::vector<double>> m;

public:
    Matrix(size_t rows, size_t cols)
    {
        this->cols = cols;
        this->rows = rows;
        this->matrix = new double *[this->cols];
        this->initAndClear();
    }

    ~Matrix()
    {
        // free(this->matrix);
    }

    double at(int i, int j);
    void set(int i, int j, double num);
    void initAndClear();
    void add(double n);
    void transpose();
    void add(Matrix n);
    void multiply(double n);
    Matrix multiply(Matrix n);
    void map(double (*activation)(double sum));


    void printMatrix();

    // DEPRECATED
    void randomise();
};

#endif
