//
// Created by otost on 05/10/2022.
//

#include "./headers/layer.h"
using namespace std;

Matrix Layer::getOutputs(Matrix inputs) const {
    return inputs.multiply(this->weights);
}