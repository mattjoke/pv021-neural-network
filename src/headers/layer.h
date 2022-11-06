//
// Created by otost on 05/10/2022.
//

#ifndef PV021_NEURAL_NETWORK_LAYER_H
#define PV021_NEURAL_NETWORK_LAYER_H

#include <utility>

#include "matrix.h"
#include "activation.h"

using namespace std;

class Layer {
public:

    size_t perceptrons_below;
    // Neurons in this layer
    Matrix neurons = Matrix(0, 0);
    // Incoming weights to this layer
    Matrix weights = Matrix(0, 0);
    // Bias for this layer
    Matrix bias = Matrix(0, 0);
    // Activation function for this layer (default: 'logistic')
    ActivationFunction activationFunction = Activation::logistic();

public:
    Layer(size_t num_perceptrons_below, size_t num_perceptrons) {
        this->perceptrons_below = num_perceptrons_below;
        this->weights = Matrix(this->perceptrons_below, num_perceptrons);
        this->neurons = Matrix(num_perceptrons, 1);
        this->bias = Matrix(num_perceptrons_below, 1);

        // Randomize weights and bias
        this->weights.randomise();
        this->bias.randomise();
    }

    Matrix feedForward(Matrix inputs);

    Matrix getWeights() const;

    void setWeights(Matrix weights);

    Matrix getBias() const;

    void setBias(Matrix bias);

    void updateBias(Matrix bias);

    void updateWeights(Matrix weights);

    void setNeurons(Matrix neurons) {
        this->neurons = neurons;
    }

    Matrix getNeurons() const {
        return this->neurons;
    }

    ActivationFunction getActivationFunction() const {
        return this->activationFunction;
    }

    void printInformation() const;
};

#endif // PV021_NEURAL_NETWORK_LAYER_H
