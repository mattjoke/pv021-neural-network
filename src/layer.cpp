//
// Created by otost on 05/10/2022.
//

#include "headers/layer.h"

using namespace std;

Matrix Layer::feedForward(Matrix inputs) {
    cout << "inputs change\n";
    this->neurons = inputs.transpose();
    outputs_from_weighted_sum = inputs.multiply(this->weights);
    outputs_from_weighted_sum.add(this->bias);
    return outputs_from_weighted_sum.map(this->activationFunction.function);
}

void Layer::setWeights(Matrix weights) {
    this->weights = weights;
}

Matrix Layer::getWeights() const {
    return this->weights;
}

Matrix Layer::getBias() const {
    return this->bias;
}

void Layer::setBias(Matrix bias) {
    this->bias = bias;
}

void Layer::updateBias(Matrix bias) {
    this->bias = bias;
}

void Layer::updateWeights(Matrix cost) {
    // Matrix c = cost.transpose();
    this->weights = cost.multiply(this->neurons.transpose());
}

void Layer::printInformation() const {
cout << "Layer information:" << endl;
    cout << "Number of perceptrons below: " << this->perceptrons_below << endl;
    cout << "Number of perceptrons: " << this->neurons.getRows() << endl;
    cout << "Weights:" << endl;
    this->weights.printMatrix();
    cout << "Bias:" << endl;
    this->bias.printMatrix();
}

Matrix Layer::getOutputsFromWeightedSum() const {
    return outputs_from_weighted_sum;
}
