#include "neural_network.h"

void NeuralNetwork::setActivationFunction(const std::string &activation) {
    Activation::parseActivationFunction(activation, &(this->activationFunction));
}

void NeuralNetwork::setInputLayerSize(size_t size) {
    this->inputLayerSize = size;
    // TODO: Recompute whole class!
}

void NeuralNetwork::setOutputLayerSize(size_t size) {
    this->outputLayerSize = size;
    // TODO: Recompute array!
}

void NeuralNetwork::setNumberOfHiddenLayers(size_t size) {
    this->numberOfHiddenLayers = size;
    // TODO: Recompute array!
}


void NeuralNetwork::printData() {
    std::cout << "Input layer size: " << this->inputLayerSize << std::endl;
    std::cout << "Output layer size: " << this->outputLayerSize << std::endl;
    std::cout << "Number of hidden layers: " << this->numberOfHiddenLayers << std::endl;
    std::cout << "Hidden layer sizes: ";
    for (size_t i = 0; i < this->numberOfHiddenLayers; i++) {
        std::cout << this->hiddenLayerSizes[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Network vector" << endl;
    for (size_t i = 0; i < this->network.size(); i++) {
        std::cout << "Layer " << i << endl;
        this->network[i].printInformation();
    }
    std::cout << std::endl;
}

void NeuralNetwork::buildNetwork() {
    this->network.clear();
    if (this->numberOfHiddenLayers == 0) {
        std::cout << "No hidden layers! The input neurons are directly connected to output!" << std::endl;
        this->network.emplace_back(this->inputLayerSize, this->outputLayerSize);
        return;
    }

    // First layer -> first hidden layer
    auto inputLayer = Layer(this->inputLayerSize, this->hiddenLayerSizes[0]);
    this->network.emplace_back(inputLayer);
    // Create other hidden layers
    for (int i = 0; i < this->numberOfHiddenLayers-1; ++i) {
        auto hiddenLayer = Layer(this->hiddenLayerSizes[i], this->hiddenLayerSizes[i + 1]);
        this->network.emplace_back(hiddenLayer);
    }
    // Last hidden layer -> output layer
    auto outputLayer = Layer(this->hiddenLayerSizes[this->numberOfHiddenLayers - 1], this->outputLayerSize);
    this->network.emplace_back(outputLayer);
}

Matrix NeuralNetwork::feedForward(const vector<double> &input) {
    // Check if the input is the same size
    if (input.size() != this->inputLayerSize){
        cout << "The input is not correct, not forwarding further" << endl;
        return {0, 0};
    }

    Matrix buffer = convertVectorToMatrix(input);
    for (auto & i : this->network) {
        buffer = i.feedForward(buffer);
    }
    return buffer;
}
