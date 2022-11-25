//
// Created by otost on 05/10/2022.
//

#include "headers/layer.h"

using namespace std;

vector<double> Layer::feedForward(vector<double> inputs) {
    // weighted sums.size() == inputs.size()
    for (int i=0; i<weightedSums.size(); i++) {
        weightedSums[i] = bias[i];
        for (int j=0; j < inputs.size(); j++) {
            weightedSums[i] += weights[j][i] * inputs[j];
        }
    }
    vector<double> outputs(weightedSums.size());
    for (int i=0; i<weightedSums.size(); i++) {
        outputs[i] = this->activationFunction.function(weightedSums[i]);
    }
//    for (double & output : outputs) {
//        output = output < 0.0000000001 ? 0 : output;
//    }

    return outputs;
}

vector<double> Layer::backPropagate(vector<double> cost) {
    return cost;
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
        bias[i] -= (this->learningRate * cost[i]);
    }
}

void Layer::updateWeights(vector<double> cost) {
    return;
}

void Layer::printInformation() const {
cout << "Layer information:" << endl;
    cout << "Number of perceptrons below: " << this->perceptrons_below << endl;
    cout << "Number of perceptrons: " << this->weightedSums.size() << endl;
    cout << "Weights:" << endl;
    for (auto row:weights) {
        for (auto elem: row) {
            cout << elem << " ";
        }
        cout << endl;
    }
    cout << "Bias:" << endl;
    for (auto elem: bias) {
        cout << elem << " " << endl;
    }
}