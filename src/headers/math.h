#include <iostream>
using namespace std;

class Matrix
{
private:
    size_t cols;
    size_t rows;
    double **matrix;

public:
    Matrix(size_t rows, size_t cols)
    {
        this->cols = cols;
        this->rows = rows;
        this->matrix = new double *[this->cols];
        this->initAndClear();
    }

    void initAndClear();
    void add(double n);
    void transpose();
    void add(Matrix n);
    void multiply(double n);
    Matrix multiply(Matrix n);

    void printMatrix();

    // DEPRECATED
    void randomise();
};
