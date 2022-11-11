//
// Created by otost on 05/10/2022.
//

#include "headers/layer.h"

using namespace std;

vector<double> Layer::feedForward(vector<double> inputs) {
    for (int i=0; i<weights.size(); i++) {
        weightedSums[i] = 0;
        for (int j=0; j < weights[0].size(); j++) {
            weightedSums[i] += weights[i][j] * inputs[j];
        }
        weightedSums[i] += bias[i];
    }

    for (int i=0; i<inputs.size(); i++) {
        inputs[i] = this->activationFunction.function(weightedSums[i]);
    }
    return inputs;
}

void Layer::setWeights(vector<vector<double>> weights) {
    this->weights = weights;
}

vector<vector<double>> Layer::getWeights() const {
    return this->weights;
}

vector<double> Layer::getBias() const {
    return this->bias;
}

void Layer::setBias(vector<double> bias) {
    this->bias = bias;
}

void Layer::updateBias(vector<double> cost) {
    for (int i=0; i<bias.size(); i++) {
        bias[i] = bias[i] - (0.1 * cost[i]);
    }
}

void Layer::updateWeights(vector<double> cost) {
    cost.multiply(0.1);
    this->weights = this->weights.sub(cost);
}

void Layer::printInformation() const {
cout << "Layer information:" << endl;
    cout << "Number of perceptrons below: " << this->perceptrons_below << endl;
    cout << "Number of perceptrons: " << this->weightedSums.getRows() << endl;
    cout << "Weights:" << endl;
    this->weights.printMatrix();
    cout << "Bias:" << endl;
    this->bias.printMatrix();
}