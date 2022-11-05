#ifndef PV021_NEURAL_NETWORK_ACTIVATION_H
#define PV021_NEURAL_NETWORK_ACTIVATION_H

#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

class Activation
{
public:
    static double identity(double sum);
    static double logistic(double sum);
    static double tanh(double sum);
    static double relu(double sum);

    static void parseActivationFunction(const string& activation, double (**activationFunction)(double sum));
};

#endif
