#include "neural_network.h"


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
        Layer l = Layer(this->inputLayerSize, this->outputLayerSize, this->learningRate);
        l.activationFunction = Activation::softmax();
        this->network.emplace_back(l);
        return;
    }

    // First layer -> first hidden layer
    auto inputLayer = Layer(this->inputLayerSize, this->hiddenLayerSizes[0], this->learningRate);
    this->network.emplace_back(inputLayer);
    // Create other hidden layers
    for (int i = 0; i < this->numberOfHiddenLayers - 1; ++i) {
        auto hiddenLayer = Layer(this->hiddenLayerSizes[i], this->hiddenLayerSizes[i + 1], this->learningRate);
        this->network.emplace_back(hiddenLayer);
    }
    // Last hidden layer -> output layer
    auto outputLayer = Layer(this->hiddenLayerSizes[this->numberOfHiddenLayers - 1], this->outputLayerSize,
                             this->learningRate);
    outputLayer.activationFunction = Activation::softmax();
    this->network.emplace_back(outputLayer);
}

vector<double> NeuralNetwork::feedForward(const vector<double> &input) {
    // Check if the input is the same size
    if (input.size() != this->inputLayerSize) {
        throw invalid_argument("NeuralNetwork::feedForward -> Input size is not the same as the input layer size!");
    }

    //Matrix buffer = convertVectorToMatrix(input);
    vector<double> buffer = input;
    for (auto &i: this->network) {
        buffer = i.feedForward(buffer);
    }
    return buffer;
}

void NeuralNetwork::backPropagation(const vector<double> &inputs, const vector<double> &targets) {
    // Check if the input is the same size
    if (inputs.size() != this->inputLayerSize) {
        throw invalid_argument("NeuralNetwork::backPropagation -> The input is not correct, not forwarding further");
    }
    // Check if the target is the same size
    if (targets.size() != this->outputLayerSize) {
        throw invalid_argument("NeuralNetwork::backPropagation -> Target is not correct, not forwarding further");
    }

    // Feed forward
    vector<double> ff = feedForward(inputs);

    // Cost derivative
    vector<double> zL = network[network.size() - 1].getWeightedSums();
    double wholeSum = 0.0;
    for (double weightedSum : network[network.size() - 1].getWeightedSums()) {
        wholeSum += exp(weightedSum);
    }
    for (int i = 0; i < zL.size(); i++) {
        zL[i] = network[network.size() - 1].activationFunction.derivative(zL[i], wholeSum);
    }
    vector<double> cost = costDerivative(ff, targets);
    for (int i = 0; i < cost.size(); i++) {
        cost[i] = cost[i] * zL[i];
    }

    // Update weights and bias
    network[network.size() - 1].updateBias(cost);
    if (network.size() == 1) {
        vector<double> t = {};
        for (int i = 0; i < inputs.size(); i++) {
            t.emplace_back(network[0].activationFunction.function(inputs[i], 1));
        }
        for (int i = 0; i < network[network.size() - 1].weights.size(); i++) {
            for (int j = 0; j < network[network.size() - 1].weights[0].size(); j++) {
                network[0].weights[i][j] -= t[i] * 0.1 * cost[j];
            }
        }
        return;
    }
    vector<double> outputsFromLowerLayer = {};
    // Apply activation function to the weighted sums
    for (int i = 0; i < network[network.size() - 2].weightedSums.size(); i++) {
        outputsFromLowerLayer.emplace_back(
                network[network.size() - 2].activationFunction.function(network[network.size() - 2].weightedSums[i], 1));
    }
    // Update weights
    for (int i = 0; i < network[network.size() - 1].weights.size(); i++) {
        for (int j = 0; j < network[network.size() - 1].weights[0].size(); j++) {
            network[network.size() - 1].weights[i][j] -= outputsFromLowerLayer[i] * this->learningRate * cost[j];
        }
    }

    // Backpropagation
    for (int i = network.size() - 2; i >= 0; i--) {
        vector<double> sp = {};
        for (int j = 0; j < network[i].weightedSums.size(); j++) {
            sp.emplace_back(network[i].activationFunction.derivative(network[i].weightedSums[j], 1));
        }
        vector<double> delta = {};

        for (int k = 0; k < network[i + 1].weights.size(); k++) {
            double num = 0;
            for (int j = 0; j < cost.size(); j++) {
                num += cost[j] * network[i + 1].weights[j][k];
            }
            delta.emplace_back(num);
        }

        cost = {};
        for (int j = 0; j < delta.size(); j++) {
            cost.emplace_back(delta[j] * sp[j]);
        }

        // Update weights and biases
        this->network[i].updateBias(cost);
        if (i == 0) {
            for (int j = 0; j < network[0].weights.size(); j++) {
                for (int k = 0; k < network[0].weights[0].size(); k++) {
                    network[0].weights[j][k] -= inputs[j] * this->learningRate * cost[k];
                }
            }
            return;
        }
        vector<double> outputsFromLowerLayer = {};
        for (int j = 0; j < network[i - 1].weightedSums.size(); j++) {
            outputsFromLowerLayer.emplace_back(network[i].activationFunction.function(network[i - 1].weightedSums[j], 1));
        }
        for (int j = 0; j < network[i].weights.size(); j++) {
            for (int k = 0; k < network[i].weights[0].size(); k++) {
                this->network[i].weights[j][k] -= outputsFromLowerLayer[j] * this->learningRate * cost[k];
            }
        }
    }
}

vector<vector<double>> NeuralNetwork::predict(const vector<vector<double>> &inputs) {
    vector<vector<double>> outputs;
    for (auto &i: inputs) {
        vector<double> prediction = predict(i);
        vector<double> output;
        for (int j = 0; j < prediction.size(); ++j) {
            output.emplace_back(prediction.at(j));
        }
        outputs.emplace_back(output);
    }
    return outputs;
}

vector<double> NeuralNetwork::predict(const vector<double> &input) {
    return feedForward(input);
}

int getIndexOfHighestValue(const vector<double> input) {
    int index = 0;
    for (int i = 0; i < input.size(); i++) {
        if (input[i] > input[index]) {
            index = i;
        }
    }
    return index;
}

vector<double> vectorHighestValue(const vector<double> input) {
    int category = getIndexOfHighestValue(input);
    vector<double> output;
    for (int i = 0; i < input.size(); ++i) {
        output.emplace_back((double) i == category);
    }
    return output;
}

void NeuralNetwork::accuracy(const vector<vector<double>> &inputs, const vector<vector<double>> &targets) {
    if (inputs.size() != targets.size()) {
        throw invalid_argument("NeuralNetwork::accuracy -> The number of inputs and targets are not the same!");
    }
    int correct = 0;
    for (int i = 0; i < inputs.size(); ++i) {
        auto resultCategory = vectorHighestValue(inputs[i]);
        if (resultCategory == targets[i]) {
            correct++;
        }
        /*else {
            cout << "Incorrect: " << i << endl;
            cout << "Result: " <<endl;
            for (int j = 0; j < resultCategory.size(); ++j) {
                cout << resultCategory[j] << " ";
            }
            cout << endl;
            cout << "Target: " <<endl;
            for (int j = 0; j < targets[i].size(); ++j) {
                cout << targets[i][j] << " ";
            }
            cout << endl;
            cout << "----------------" << endl;
        }*/
    }
    cout << "Accuracy: " << (double) correct / inputs.size() << endl;
    cout << "Correct: " << correct << "/" << inputs.size() << endl;
}

void NeuralNetwork::train(const vector<vector<double>> &inputs, const vector<vector<double>> &targets,
                          size_t batchSize, size_t epochs) {
    if (inputs.size() != targets.size()) {
        throw invalid_argument("NeuralNetwork::train -> The number of inputs and targets are not the same!");
    }
    for (int i = 0; i < epochs; ++i) {
        cout << "Epoch " << i << endl;
        for (size_t j = 0; j < inputs.size(); j += batchSize) {
            vector<vector<double>> batchInputs;
            vector<vector<double>> batchTargets;
            for (size_t k = 0; k < batchSize; ++k) {
                if (j + k >= inputs.size()) {
                    break;
                }
                batchInputs.emplace_back(inputs[j + k]);
                batchTargets.emplace_back(targets[j + k]);
            }
            trainBatch(batchInputs, batchTargets);
        }
        cout << "Accuracy: " << endl;
        auto predictions = predict(inputs);
        accuracy(predictions, targets);
    }
}

void NeuralNetwork::trainBatch(const vector<vector<double>> &inputs, const vector<vector<double>> &targets) {
    if (inputs.size() != targets.size()) {
        throw invalid_argument("NeuralNetwork::trainBatch -> The number of inputs and targets are not the same!");
    }
    for (int i = 0; i < inputs.size(); ++i) {
        backPropagation(inputs[i], targets[i]);
    }
}

