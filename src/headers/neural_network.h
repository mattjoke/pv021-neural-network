//
// Created by Matej Hakoš on 11/4/2022.
//

#ifndef PV021_NEURAL_NETWORK_NEURAL_NETWORK_H
#define PV021_NEURAL_NETWORK_NEURAL_NETWORK_H

#include <iostream>
#include "activation.h"
#include "utils.h"

using namespace std;

class NeuralNetwork {
private:
    // Layer sizes
    size_t inputLayerSize;

    size_t outputLayerSize;

    // Number of hidden layers is provided by the size of the vector
    size_t numberOfHiddenLayers;
    vector<size_t> hiddenLayerSizes;

    // Learning rate
    double learningRate = 0.15;


    //Network data
    vector<Layer> network;

    // Activation function (default: 'relu')
    ActivationFunction activationFunction;

public:
    NeuralNetwork(size_t inputLayerSize, size_t outputLayerSize, const vector<size_t> &hiddenLayerSizes) {
        this->inputLayerSize = inputLayerSize;
        this->outputLayerSize = outputLayerSize;
        this->hiddenLayerSizes = hiddenLayerSizes;
        this->numberOfHiddenLayers = hiddenLayerSizes.size();
        this->activationFunction = Activation::relu();

        buildNetwork();
    }

    void buildNetwork();

    vector<double> feedForward(const vector<double> &input);

    void backPropagation(const vector<double> &inputs, const vector<double> &targets);

    void train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets);

    Matrix predict(const vector<double> &inputs);

    vector<vector<double>> predict(const vector<vector<double>> &inputs);

   static void accuracy(const vector<vector<double>> &inputs, const vector<vector<double>> &targets);

    void setActivationFunction(const string &activation = "relu");

    void setInputLayerSize(size_t size);

    void setOutputLayerSize(size_t size);

    void setNumberOfHiddenLayers(size_t size);

    void printData();

    static vector<double> costDerivative(vector<double> outputActivations, const vector<double> &y) {
        vector<double> result = {};
        for (int i=0; i<outputActivations.size(); i++) {
            result.emplace_back(outputActivations[i] - y[i]);
        }
        return result;
    }
};


#endif //PV021_NEURAL_NETWORK_NEURAL_NETWORK_H
