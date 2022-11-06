//
// Created by otost on 05/10/2022.
//

#include "headers/layer.h"

using namespace std;

Matrix Layer::feedForward(Matrix inputs) {
    inputs.multiply(this->weights);
    inputs.add(this->bias);
    this->weightedSums = inputs;
    inputs.mapSelf(this->activationFunction.function);
    return inputs;
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

void Layer::updateBias(Matrix cost) {
    cost.multiply(0.1);
    this->bias = this->bias.sub(cost);
}

void Layer::updateWeights(Matrix cost) {
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