//
// Created by Matej Hakoš on 11/4/2022.
//

#ifndef PV021_NEURAL_NETWORK_UTILS_H
#define PV021_NEURAL_NETWORK_UTILS_H

#include "matrix.h"

Matrix convertVectorToMatrix(vector<double> vec) {
    Matrix m(1, vec.size());
    for (int i = 0; i < vec.size(); i++) {
        m.set(0, i, vec[i]);
    }
    return m;
}

#endif //PV021_NEURAL_NETWORK_UTILS_H
