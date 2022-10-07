#ifndef PV021_NEURAL_NETWORK_ACTIVATION_H
#define PV021_NEURAL_NETWORK_ACTIVATION_H

using namespace std;

class Activation
{
public:
    static double identity(double sum);
    static double logistic(double sum);
    static double tanh(double sum);
    static double relu(double sum);

    static void parseActivationFunction(string activation, double (**activationFunction)(double sum));
};

#endif
