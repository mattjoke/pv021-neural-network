//
// Created by otost on 05/10/2022.
//

#ifndef PV021_NEURAL_NETWORK_LAYER_H
#define PV021_NEURAL_NETWORK_LAYER_H

#include "utils.h"

using namespace std;

class Layer {
public:
    size_t percepts_below;
    size_t num_perceptrons;
    Matrix weights = Matrix(0, 0);

public:
    Layer(size_t num_perceptrons_below, size_t num_perceptrons) {
        this->percepts_below = num_perceptrons_below;
        this->num_perceptrons = num_perceptrons;
        this->weights = Matrix(num_perceptrons_below, num_perceptrons);
    }

    Matrix getOutputs(Matrix inputs) const;
};


#endif //PV021_NEURAL_NETWORK_LAYER_H
