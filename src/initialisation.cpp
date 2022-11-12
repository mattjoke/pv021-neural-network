//
// Created by Matej Hakoš on 11/11/2022.
//

#include "initialisation.h"
#include <random>
using namespace std;



void Initialisation::leCun(double favIn, vector<vector<double>> *weights) {
    random_device rd{};
    mt19937 generator{rd()};

    std::normal_distribution<double> distribution(0.0, favIn);
    for (auto & i : *weights) {
        for (double & j : i) {
            j = distribution(generator);
        }
    }
};

void Initialisation::xavier(double favAvg, vector<vector<double>> *weights) {
    leCun(favAvg, weights);
}

void Initialisation::he(double favIn, vector<vector<double>> *weights) {
    leCun(favIn, weights);
}

void Initialisation::zero(vector<vector<double>> *weights) {
    leCun(0, weights);
}

void Initialisation::one(vector<vector<double>> *weights) {
    for (int i= 0; i < weights->size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            (*weights)[i][j] =  1;
        }
    }
}
