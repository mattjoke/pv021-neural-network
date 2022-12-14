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
    vector<double> weightedSums = {};
    // Incoming weights to this layer
    vector<vector<double>> weights = {};
    // Bias for this layer
    vector<double> bias = {};
    // Activation function for this layer (default: 'relu')
    ActivationFunction activationFunction = Activation::logistic();

public:
    Layer(size_t num_perceptrons_below, size_t num_perceptrons) {
        this->perceptrons_below = num_perceptrons_below;
        this->weights = vector<vector<double>>(this->perceptrons_below, vector<double>(num_perceptrons, 0.0)); // this->perceptrons_below, num_perceptrons

        this->weightedSums = vector<double>(num_perceptrons); // num_perceptrons, 1
        this->bias = vector<double>(num_perceptrons, 0.001); // 1, num_perceptrons

        // Randomize weights
        Initialisation::he(2.0 / sqrt(this->perceptrons_below), &this->weights);
    }

    vector<double> feedForward(vector<double> inputs);

    vector<vector<double>> getWeights() const;

    void setWeights(vector<vector<double>> weights);

    vector<double> getBias() const;

    void setBias(vector<double> bias);

    void updateBias(vector<double> cost);

    //void updateWeights(Matrix cost);

//    void getWeightedSums(Matrix neurons) {
//        this->weightedSums = neurons;
//    }

    vector<double> getWeightedSums() const {
        return weightedSums;
    }

    ActivationFunction getActivationFunction() const {
        return this->activationFunction;
    }

    void printInformation() const;
};

#endif // PV021_NEURAL_NETWORK_LAYER_H
