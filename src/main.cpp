#include <iostream>
#include "headers/utils.h"
#include "headers/layer.h"
using namespace std;

int main()
{
    // Perceptron *p = new Perceptron(2);
    // static double ar[3][3] = {{-1, 0, 1}, {0, 1, 1}, {3, 0, 0}};

    // for (size_t i = 0; i < 3; i++)
    // {
    //     double *array = ar[i];
    //     double input[2] = {array[0], array[1]};
    //     p->train(input, array[2]);
    //     p->printWeigthVector();
    // }
    // cout << "End of trainning\n";
    // p->printWeigthVector();
    // double point[2] = {-1, 0};
    // // cout << p->feedForward(point) << "\n";
    // double point2[2] = {0, 1};
    // // cout << p->feedForward(point2) << "\n";
    // double point3[2] = {3, 0};
    // // cout << p->feedForward(point3) << "\n";
    // NeuralNetwork *n = new NeuralNetwork();
    Matrix *m = new Matrix(2, 3);
    m->add(+1);
    // m->printMatrix();

    Matrix *n = new Matrix(1, 2);
    n->randomise();
    n->printMatrix();

    n->transpose();
    // n->printMatrix();

    n->transpose();
    // n->printMatrix();
    // n->multiply(*m);
    // n->printMatrix();

    Layer *l = new Layer(2, 3);
    l->getWeights().printMatrix();
    l->getBias().printMatrix();
    *n = l->feedForward(*n);
    n->printMatrix();
}
