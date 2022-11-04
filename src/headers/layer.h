//
// Created by otost on 05/10/2022.
//

#ifndef PV021_NEURAL_NETWORK_LAYER_H
#define PV021_NEURAL_NETWORK_LAYER_H

#include "matrix.h"
#include "activation.h"
using namespace std;

class Layer
{
public:

    size_t percepts_below;
    size_t num_perceptrons;
    Matrix weights = Matrix(0,0);
    Matrix bias = Matrix(0, 0);
    double (*activationFunction)(double sum) = Activation::logistic;

public:
    Layer(size_t num_perceptrons_below, size_t num_perceptrons)
    {
        this->percepts_below = num_perceptrons_below;
        this->num_perceptrons = num_perceptrons;
        this->weights = Matrix(this->percepts_below, this->num_perceptrons);
        this->bias = Matrix(1, num_perceptrons);
        this->weights.randomise();
        this->bias.randomise();
    }

    Matrix feedForward(Matrix inputs) const;
    Matrix getWeights() const;
    void setWeights(Matrix weights);
    Matrix getBias() const;
    void setBias(Matrix bias);

    void printInformation() const;
};

#endif // PV021_NEURAL_NETWORK_LAYER_H
