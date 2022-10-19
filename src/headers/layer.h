//
// Created by otost on 05/10/2022.
//

#ifndef PV021_NEURAL_NETWORK_LAYER_H
#define PV021_NEURAL_NETWORK_LAYER_H

#include "utils.h"
#include "activation.h"
using namespace std;

class Layer
{
private:
    size_t percepts_below;
    size_t num_perceptrons;
    Matrix weights = Matrix(percepts_below, num_perceptrons);
    Matrix bias = Matrix(0, 0);
    double (*activationFunction)(double sum) = *Activation::relu;

public:
    Layer(size_t num_perceptrons_below, size_t num_perceptrons)
    {
        this->percepts_below = num_perceptrons_below;
        this->num_perceptrons = num_perceptrons;
        this->weights = Matrix(num_perceptrons_below, num_perceptrons);
        this->bias = Matrix(1, num_perceptrons);
        this->weights.randomise();
        this->bias.randomise();
    }

    Matrix feedForward(Matrix inputs);
    void setWeights(Matrix weights)
    {
        this->weights = weights;
    }
    Matrix getWeights()
    {
        return this->weights;
    }
    Matrix getBias()
    {
        return this->bias;
    }
};

#endif // PV021_NEURAL_NETWORK_LAYER_H
