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
    size_t num_perceptrons;
    // Neurons in this layer
    vector<double> weightedSums = {};
    // Incoming weights to this layer
    vector<vector<double>> weights = {};
    // Bias for this layer
    vector<double> bias = {};
    // Activation function for this layer (default: 'relu')
    ActivationFunction activationFunction = Activation::relu();

    // Learning rate for this layer
    double learningRate;

public:
    Layer(size_t num_perceptrons_below, size_t num_perceptrons, double learningRate) {
        this->perceptrons_below = num_perceptrons_below;
        this->num_perceptrons = num_perceptrons;
        this->weights = vector<vector<double>>(this->perceptrons_below, vector<double>(num_perceptrons, 0.0)); // this->perceptrons_below, num_perceptrons

        clearBeforeBatch();
        this->bias = vector<double>(num_perceptrons, 0.01);
        this->learningRate = learningRate;

        // Randomize weights
        Initialisation::he(2.0 / sqrt(this->perceptrons_below), &this->weights);
        // Randomize bias
    }

    void clearBeforeBatch();

    vector<double> feedForward(vector<double> inputs);

    void updateWeightsAndBiases(vector<double> cost, vector<double> outputsFromLowerLayer);

    void printInformation() const;
};

#endif // PV021_NEURAL_NETWORK_LAYER_H
