//
// Created by otost on 05/10/2022.
//

#ifndef PV021_NEURAL_NETWORK_LAYER_H
#define PV021_NEURAL_NETWORK_LAYER_H

#include <utility>

#include "matrix.h"
#include "activation.h"
#include "initialisation.h"

using namespace std;

class Layer {
public:

    size_t perceptrons_below;
    // Neurons in this layer
    Matrix weightedSums = Matrix(0, 0);
    // Incoming weights to this layer
    Matrix weights = Matrix(0, 0);
    // Bias for this layer
    Matrix bias = Matrix(0, 0);
    // Activation function for this layer (default: 'relu')
    ActivationFunction activationFunction = Activation::logistic();

public:
    Layer(size_t num_perceptrons_below, size_t num_perceptrons) {
        this->perceptrons_below = num_perceptrons_below;
        this->weights = Matrix(this->perceptrons_below, num_perceptrons);

        this->weightedSums = Matrix(num_perceptrons, 1);
        this->bias = Matrix(1, num_perceptrons);

        // Randomize weights and bias
        Initialisation::he(2.0 / sqrt(this->perceptrons_below), &this->weights);
        Initialisation::zero(&this->bias);
        this->bias.add(0.001);
    }

    Matrix feedForward(Matrix inputs);

    Matrix getWeights() const;

    void setWeights(Matrix weights);

    Matrix getBias() const;

    void setBias(Matrix bias);

    void updateBias(Matrix cost);

    void updateWeights(Matrix cost);

    void getWeightedSums(Matrix neurons) {
        this->weightedSums = neurons;
    }

    Matrix getWeightedSums() const {
        return this->weightedSums;
    }

    ActivationFunction getActivationFunction() const {
        return this->activationFunction;
    }

    void printInformation() const;
};

#endif // PV021_NEURAL_NETWORK_LAYER_H
