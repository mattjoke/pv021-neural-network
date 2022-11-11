//
// Created by Matej Hakoš on 11/11/2022.
//

#ifndef PV021_NEURAL_NETWORK_INITIALISATION_H
#define PV021_NEURAL_NETWORK_INITIALISATION_H

#include <random>
#include "matrix.h"

class Initialisation {
public:
    std::mt19937 gen;
public:
    static void xavier(double favAvg, Matrix *weights);
    static void he(double favIn, Matrix *weights);
    static void leCun(double favIn, Matrix *weights);
    static void zero(Matrix *weights);
    static void one(Matrix *weights);
};


#endif //PV021_NEURAL_NETWORK_INITIALISATION_H
