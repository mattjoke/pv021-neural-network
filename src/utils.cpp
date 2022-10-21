#include <iostream>
#include "headers/utils.h"

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

vector<vector<double>> init_matrix(int rows, int cols) {
    vector<vector<double>> vec(rows);
    for(int i=0; i <rows; i++) {
        vector<double> row(cols);
        for(int j=0; j<cols; j++) {
            row[j] = 0;
        }
        vec[i] = row;
    }
    return vec;
}

void Matrix::add(double n)
{
    for (size_t i = 0; i < this->rows; i++)
    {
        for (size_t j = 0; j < this->cols; j++)
        {
            this->m[i][j] += n;
        }
    }
}

void Matrix::add(Matrix n)
{
    if (rows != n.rows || cols != n.cols)
    {
        return;
    }
    for (size_t i = 0; i < this->rows; i++)
    {
        for (size_t j = 0; j < this->cols; j++)
        {
            this->m[i][j] += n.m[i][j];

        }
    }
}

void Matrix::multiply(double n)
{
    for (size_t i = 0; i < this->rows; i++)
    {
        for (size_t j = 0; j < this->cols; j++)
        {
            this->m[i][j] *= n;
        }
    }
}

// A.K.A. Cross-Product
Matrix Matrix::multiply(Matrix n)
{
    if (this->cols != n.rows)
    {
        invalid_argument("Left Matrix should have the same rows as the columns in the Right Matrix");
        return Matrix(0, 0);

    }
    auto *p = new Matrix(this->rows, n.cols);
    for (size_t i = 0; i < p->rows; i++)
    {
        for (size_t j = 0; j < p->cols; j++)
        {
            for (size_t k = 0; k < this->cols; k++)
            {
                //cout << this->matrix[i][k] << " " << n.matrix[k][j] << endl;
                p->m[i][j] += this->m[i][k] * n.m[k][j];
            }
            //cout << endl;
        }
    }
    // why the next 3 lines?
    this->cols = p->cols;
    this->rows = p->rows;
    this->m = p->m;
    return *p;
}

void Matrix::transpose()
{
    auto *newMatrix = new Matrix(this->cols, this->rows);
    for (size_t i = 0; i < this->rows; i++)
    {
        for (size_t j = 0; j < this->cols; j++)
        {
            newMatrix->m[j][i] = this->m[i][j];
        }
    }
    this->cols = newMatrix->cols;
    this->rows = newMatrix->rows;
    this->m = newMatrix->m;
}

void Matrix::printMatrix()
{
    cout << "Matrix with the size: (" << this->rows << "," << this->cols << ")\n";
    for (size_t i = 0; i < this->rows; i++)
    {
        for (size_t j = 0; j < this->cols; j++)
        {
            cout << "|" << this->m[i][j] << "\t";
        }
        cout << "\n";
    }
    cout << "\n";
}

void Matrix::map(double (*activation)(double sum))
{
    for (size_t i = 0; i < this->rows; i++)
    {
        for (size_t j = 0; j < this->cols; j++)
        {
            this->m[i][j] = activation(this->m[i][j]);
        }
    }
}


// DEPRECATED
void Matrix::randomise()
{
    for (size_t i = 0; i < this->rows; i++)
    {
        for (size_t j = 0; j < this->cols; j++)
        {
            this->m[i][j] = std::rand() % 10;
        }
    }
}


double Matrix::at(int i, int j) {
    return this->m[i][j];
}

void Matrix::set(int i, int j, double num) {
    this->m[i][j] = num;
}
