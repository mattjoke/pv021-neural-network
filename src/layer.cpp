//
// Created by otost on 05/10/2022.
//

#include "headers/layer.h"
#include "float.h"

using namespace std;


void Layer::clearBeforeBatch() {
    // Inputs are emplaced back, so we need to clear them before each batch
    inputs = {};
    weightedSums = {};
    activatedWeightedSums = {};
}

vector<double> Layer::feedForward(vector<double> inputs) {
    // weighted sums.size() == inputs.size()
    this->inputs.emplace_back(inputs);
    vector<double> tmpWeightedSums(num_perceptrons);
    for (int i = 0; i < num_perceptrons; i++) {
        tmpWeightedSums[i] = bias[i];
        for (int j = 0; j < inputs.size(); j++) {
            double item = tmpWeightedSums[i] + weights[j][i] * inputs[j];
            tmpWeightedSums[i] = /*item < DBL_MIN ? 0 :*/ item;
        }
    }
    weightedSums.emplace_back(tmpWeightedSums);

    vector<double> tmpActivatedWeightedSums(num_perceptrons);
    for (int i = 0; i < num_perceptrons; i++) {
        tmpActivatedWeightedSums[i] = this->activationFunction.function(tmpWeightedSums[i]);
    }
    this->activatedWeightedSums.emplace_back(tmpActivatedWeightedSums);
    return tmpActivatedWeightedSums;
}

void Layer::updateWeightsAndBiases(vector<double> cost, vector<double> outputsFromLowerLayer) {
    // Update bias
    for (int i = 0; i < bias.size(); i++) {
        bias[i] -= (this->learningRate * cost[i]);
    }
    // Update weights
    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[0].size(); j++) {
            weights[i][j] -= outputsFromLowerLayer[i] * this->learningRate * cost[j];
        }
    }
}

void Layer::printInformation() const {
    cout << "Layer information:" << endl;
    cout << "Number of perceptrons below: " << this->perceptrons_below << endl;
    cout << "Number of perceptrons: " << this->weightedSums.size() << endl;
    cout << "Weights:" << endl;
    for (auto row: weights) {
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