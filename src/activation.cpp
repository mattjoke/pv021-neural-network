#include <iostream>
#include <algorithm>
#include <cmath>
#include "headers/activation.h"
using namespace std;

double Activation::identity(double sum)
{
    return sum;
}

double Activation::logistic(double sum)
{
    return 1 / (1 + exp(-sum));
}

double Activation::tanh(double sum)
{
    return ::tanh(sum);
}

double Activation::relu(double sum)
{
    return sum >= 0 ? sum : 0;
}

void Activation::parseActivationFunction(const string& activation, double (**activationFunction)(double sum))
{
    *activationFunction = Activation::relu;
    if (activation == "identity")
    {
        *activationFunction = Activation::identity;
    }
    if (activation == "logistic")
    {
        *activationFunction = Activation::logistic;
    }
    if (activation == "tanh")
    {
        *activationFunction = Activation::tanh;
    }
}
