//
// Created by Matej Hakoš on 11/11/2022.
//

#include "initialisation.h"
#include <random>
using namespace std;



void Initialisation::leCun(double favIn, Matrix *weights) {
    random_device rd{};
    mt19937 generator{rd()};

    std::normal_distribution<double> distribution(0.0, favIn);
    for (int i= 0; i < weights->getRows(); i++) {
        for (int j = 0; j < weights->getCols(); j++) {
            weights->set(i, j, distribution(generator));
        }
    }
};

void Initialisation::xavier(double favAvg, Matrix *weights) {
    leCun(favAvg, weights);
}

void Initialisation::he(double favIn, Matrix *weights) {
    leCun(favIn, weights);
}

void Initialisation::zero(Matrix *weights) {
    leCun(0, weights);
}

void Initialisation::one(Matrix *weights) {
    for (int i= 0; i < weights->getRows(); i++) {
        for (int j = 0; j < weights->getCols(); j++) {
            weights->set(i, j, 1);
        }
    }
}
