//
// Created by otost on 05/10/2022.
//

#include "headers/layer.h"
using namespace std;

Matrix Layer::feedForward(Matrix inputs)
{
    inputs = (inputs.multiply(this->weights));
    inputs.add(this->bias);
    inputs.map(this->activationFunction);
    return inputs;
}

Matrix Layer::getOutputs(Matrix inputs) const {
    return inputs.multiply(this->weights);
}
