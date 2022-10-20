#include <iostream>
#include "headers/utils.h"
#include "headers/layer.h"
#include "perceptron.cpp"
#include "neural_network.cpp"

using namespace std;

int main()
{
    auto *m = new Matrix(2, 3);
    m->add(+1);
    // m->printMatrix();

    auto *n = new Matrix(1, 2);

    n->randomise();
    n->printMatrix();

    n->transpose();
    // n->printMatrix();

    n->transpose();
    // n->printMatrix();
    // n->multiply(*m);
    // n->printMatrix();
    cout << "input\n";
    n->printMatrix();
    auto *l = new Layer(2, 3);
    cout << "weights\n";
    l->getWeights().printMatrix();
    cout << "bias\n";
    l->getBias().printMatrix();
    *n = l->feedForward(*n);
    n->printMatrix();
    
    // Oto
    m->printMatrix();
    n->printMatrix();
    // n->multiply(*m);
    // n->printMatrix();
    m->multiply(*n);
    m->printMatrix();
    n->printMatrix();

    cout << "-----" << endl;
    auto *f = new Matrix(1, 3);
    f->set(0, 0, 5);
    f->set(0, 1, 2);
    f->set(0, 2, 3);
    f->printMatrix();
    auto* first = new Layer(3, 2);
    first->weights.set(0, 0, 1);
    first->weights.set(0, 1, 1);
    first->weights.set(1, 0, 1);
    first->weights.set(1, 1, 1);
    first->weights.set(2, 0, 1);
    first->weights.set(2, 1, 1);
    first->weights.printMatrix();
    auto* second = new Layer(2, 1);
    second->weights.set(0, 0, 1.5);
    second->weights.set(1, 0, 2);
    second->weights.printMatrix();
    second->getOutputs(first->getOutputs(*f)).printMatrix();
}
