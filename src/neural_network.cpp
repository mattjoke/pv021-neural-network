#include <iostream>
#include "activation.cpp"
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
    // Activation function (default: 'relu')
    double (*activationFunction)(double sum);
    // Solver function
    double (*solverFunction)(double sum);

    // Private functions

public:
    NeuralNetwork(size_t hiddenLayersCount = 1)
    {
        this->numberOfHiddenLayers = hiddenLayersCount;
        this->hiddenLayerSizes = new size_t[this->numberOfHiddenLayers];
        this->hiddenLayerSizes[0] = 100;
        this->activationFunction = Activation::relu;
    }

    void setActivationFunction(string activation = "relu")
    {
        Activation::parseActivationFuction(activation, &(this->activationFunction));
    }
    void setInputLayerSize(size_t size)
    {
        this->inputLayerSize = size;
        // TODO: Recompute array!
    }
    void setOutpuLayerSize(size_t size)
    {
        this->outputLayerSize = size;
        // TODO: Recompute array!
    }
    void setNumberOfHiddenLayers(size_t size)
    {
        this->numberOfHiddenLayers = size;
        this->hiddenLayerSizes = new size_t[this->numberOfHiddenLayers];
    }
    ~NeuralNetwork()
    {
        free(this->hiddenLayerSizes);
    }
};
