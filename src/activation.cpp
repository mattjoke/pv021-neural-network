#include "activation.h"

using namespace std;


ActivationFunction Activation::identity() {
    ActivationFunction identity{};
    identity.function = [](double sum, double wholeSum) { return sum; };
    identity.derivative = [](double sum, double WholeSum) { return 1.0; };
    return identity;
}

ActivationFunction Activation::logistic() {
    ActivationFunction logistic{};
    logistic.function = [](double sum, double wholeSum) { return 1.0 / (1.0 + exp(-sum)); };
    logistic.derivative = [](double sum, double wholeSum) {
        return exp(-sum) / pow(1.0 + exp(-sum), 2);
    };
    return logistic;
}

ActivationFunction Activation::tanh() {
    ActivationFunction tanh{};
    tanh.function = [](double sum, double wholeSum) { return ::tanh(sum); };
    tanh.derivative = [](double sum, double wholeSum) {
        return 1.0 - pow(::tanh(sum), 2);
    };
    return tanh;
}

ActivationFunction Activation::relu() {
    ActivationFunction relu{};
    relu.function = [](double sum, double wholeSum) { return max(0.0, sum); };
    relu.derivative = [](double sum, double wholeSum) { return sum > 0 ? 1.0 : 0.0; };
    return relu;
}

ActivationFunction Activation::softmax() {
    ActivationFunction softmax{};
    softmax.function = [](double sum, double wholeSum) {
        return exp(sum) / wholeSum;
    };
    softmax.derivative = [](double sum, double wholeSum) {
        return exp(sum) * (wholeSum - exp(sum)) / pow(wholeSum, 2);
    };
    return softmax;
}

ActivationFunction Activation::parseActivationFunction(const string &activation) {
    if (activation == "identity") {
        return Activation::identity();
    }
    if (activation == "logistic") {
        return Activation::logistic();
    }
    if (activation == "tanh") {
        return Activation::tanh();
    }
    return Activation::relu();
}


