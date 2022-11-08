#include "neural_network.h"

void NeuralNetwork::setActivationFunction(const std::string &activation) {
    this->activationFunction = Activation::parseActivationFunction(activation);
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
        std::cout << "No hidden layers! The input weightedSums are directly connected to output!" << std::endl;
        Layer l = Layer(this->inputLayerSize, this->outputLayerSize);
        l.activationFunction = Activation::softmax();
        this->network.emplace_back(l);
        return;
    }

    // First layer -> first hidden layer
    auto inputLayer = Layer(this->inputLayerSize, this->hiddenLayerSizes[0]);
    this->network.emplace_back(inputLayer);
    // Create other hidden layers
    for (int i = 0; i < this->numberOfHiddenLayers - 1; ++i) {
        auto hiddenLayer = Layer(this->hiddenLayerSizes[i], this->hiddenLayerSizes[i + 1]);
        this->network.emplace_back(hiddenLayer);
    }
    // Last hidden layer -> output layer
    auto outputLayer = Layer(this->hiddenLayerSizes[this->numberOfHiddenLayers - 1], this->outputLayerSize);
    outputLayer.activationFunction = Activation::softmax();
    this->network.emplace_back(outputLayer);
}

Matrix NeuralNetwork::feedForward(const vector<double> &input) {
    // Check if the input is the same size
    if (input.size() != this->inputLayerSize) {
        cout << "The input is not correct, not forwarding further" << endl;
        return {0, 0};
    }

    Matrix buffer = convertVectorToMatrix(input);
    for (auto &i: this->network) {
        buffer = i.feedForward(buffer);
    }
    return buffer;
}

void NeuralNetwork::backPropagation(const vector<double> &inputs, const vector<double> &targets) {
    // Check if the input is the same size
    if (inputs.size() != this->inputLayerSize) {
        throw invalid_argument("The input is not correct, not forwarding further");
    }
    // Check if the target is the same size
    if (targets.size() != this->outputLayerSize) {
        throw invalid_argument("Target is not correct, not forwarding further");
    }

    // Feed forward
    Matrix ff = feedForward(inputs);
    Matrix target = convertVectorToMatrix(targets);
    Matrix input = convertVectorToMatrix(inputs);

    // Cost derivative
    Matrix zL = this->network[this->network.size() - 1].getWeightedSums();
    zL.mapSelf(this->activationFunction.derivative);
    Matrix cost = costDerivative(ff, target).hadamard(zL);

    // Update weights and bias
    this->network[this->network.size() - 1].updateBias(cost);
    if (this->network.size() == 1) {
        auto t = input.map(this->activationFunction.function).transpose();
        this->network[this->network.size() - 1].updateWeights(t.multiply(cost));
        return;
    }
    auto o = this->network[this->network.size() - 2].getWeightedSums().map(this->activationFunction.function);
    this->network[this->network.size() - 1].updateWeights(o.transpose().multiply(cost));

    // Backpropagation
    for (int i = this->network.size() - 2; i >= 0; i--) {
        Matrix sp = this->network[i].getWeightedSums().map(this->activationFunction.derivative);
        Matrix delta = cost.multiply(this->network[i + 1].getWeights().transpose());
        cost = delta.hadamard(sp);

        // Update weights and biases
        this->network[i].updateBias(cost);
        if (i == 0) {
            auto t = input.map(this->activationFunction.function).transpose();
            this->network[i].updateWeights(t.multiply(cost));
            return;
        } else {
            this->network[i].updateWeights(this->network[i - 1].getWeightedSums().map(
                    this->activationFunction.function).transpose().multiply(cost));
        }
    }
}

vector<vector<double>> NeuralNetwork::predict(const vector<vector<double>> &inputs) {
    vector<vector<double>> outputs;
    for (auto &i: inputs) {
        Matrix prediction = predict(i);
        vector<double> output;
        for (int j = 0; j < prediction.getCols(); ++j) {
            output.emplace_back(prediction.at(0, j));
        }
        outputs.emplace_back(output);
    }
    return outputs;
}

Matrix NeuralNetwork::predict(const vector<double> &input) {
    return feedForward(input);
}

void NeuralNetwork::train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets) {
    for (int i = 0; i < inputs.size(); ++i) {
        backPropagation(inputs[i], targets[i]);
    }
}

void NeuralNetwork::accuracy(const vector<vector<double>> &inputs, const vector<vector<double>> &targets) {
    int correct = 0;
    for (int i = 0; i < inputs.size(); ++i) {
        Matrix difference = convertVectorToMatrix(inputs[i]).sub(convertVectorToMatrix(targets[i]));
        if (abs(difference.sum()) <= 0.000000000001) {
            correct++;
        }
    }
    cout << "Accuracy: " << (double) correct / inputs.size() << endl;
}
