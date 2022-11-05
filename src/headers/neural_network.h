//
// Created by Matej Hakoš on 11/4/2022.
//

#ifndef PV021_NEURAL_NETWORK_NEURAL_NETWORK_H
#define PV021_NEURAL_NETWORK_NEURAL_NETWORK_H

#include <iostream>
#include "activation.h"
#include "utils.h"
using namespace std;

class NeuralNetwork
{
private:
    // Layer sizes
    size_t inputLayerSize;

    size_t outputLayerSize;

    // Number of hidden layers is provided by the size of the vector
    size_t numberOfHiddenLayers;
    vector<size_t> hiddenLayerSizes;


    //Network data
    vector<Layer> network;

    // Activation function (default: 'relu')
    double (*activationFunction)(double sum);
    // Solver function
    double (*solverFunction)(double sum);

public:
    NeuralNetwork(size_t inputLayerSize, size_t outputLayerSize, const vector<size_t>& hiddenLayerSizes)
    {
        this->inputLayerSize = inputLayerSize;
        this->outputLayerSize = outputLayerSize;
        this->hiddenLayerSizes = hiddenLayerSizes;
        this->numberOfHiddenLayers = hiddenLayerSizes.size();
        this->activationFunction = Activation::relu;

        buildNetwork();
    }

    void buildNetwork();
    Matrix feedForward(const vector<double>& input);
    void setActivationFunction(const string& activation = "relu");
    void setInputLayerSize(size_t size);
    void setOutputLayerSize(size_t size);
    void setNumberOfHiddenLayers(size_t size);
    void printData();
};


#endif //PV021_NEURAL_NETWORK_NEURAL_NETWORK_H
