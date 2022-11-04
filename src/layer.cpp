//
// Created by otost on 05/10/2022.
//

#include "headers/layer.h"

using namespace std;

Matrix Layer::feedForward(Matrix inputs) const {
    inputs = (inputs.multiply(this->weights));
    inputs.add(this->bias);
    inputs.map(this->activationFunction);
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

void Layer::printInformation() const {
cout << "Layer information:" << endl;
    cout << "Number of perceptrons below: " << this->percepts_below << endl;
    cout << "Number of perceptrons: " << this->num_perceptrons << endl;
    cout << "Weights:" << endl;
    this->weights.printMatrix();
    cout << "Bias:" << endl;
    this->bias.printMatrix();
}