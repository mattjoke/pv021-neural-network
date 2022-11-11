#ifndef PV021_NEURAL_NETWORK_ACTIVATION_H
#define PV021_NEURAL_NETWORK_ACTIVATION_H

#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

struct ActivationFunction {
    double (*function)(double sum);
    double (*derivative)(double sum);
};

class Activation
{
public:
    static ActivationFunction identity();
    static ActivationFunction logistic();
    static ActivationFunction tanh();
    static ActivationFunction relu();

    // static ActivationFunction softmax();

    static ActivationFunction parseActivationFunction(const string& activation);
};

#endif
