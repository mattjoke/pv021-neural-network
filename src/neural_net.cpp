#include <iostream>
#include "activation_functions.cpp"
using namespace std;

class NeuralNetwork
{
private:
    // Layer sizes
    size_t inputLayerSize;
    size_t outputLayerSize;
    size_t *hiddenLayerSizes;
    // Number of hidden layers
    size_t numberOfHiddenLayers;
    // Activation function
    ActivationFunction *function;

public:
    NeuralNetwork(size_t hiddenLayersCount = 1)
    {
        this->numberOfHiddenLayers = hiddenLayersCount;
        this->hiddenLayerSizes = new size_t[this->numberOfHiddenLayers];
        this->function = ActivationFunction::identity;
    }
    ~NeuralNetwork()
    {
        free(this->hiddenLayerSizes);
    }
};
