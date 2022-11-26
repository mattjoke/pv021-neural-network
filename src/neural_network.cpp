#include "neural_network.h"

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
        // Should not happen
        std::cout << "No hidden layers! The input weightedSums are directly connected to output!" << std::endl;
        Layer l = Layer(this->inputLayerSize, this->outputLayerSize, this->learningRate);
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

    // First layer -> Last Hidden Layer
    vector<double> buffer = {};
    for (double elem : input) {
        buffer.emplace_back(elem);
    }
    for (int i = 0; i < this->network.size() - 1; ++i) {
        buffer = this->network[i].feedForward(buffer);
    }

    // Last hidden layer -> Softmax
    //Layer* lastLayer = &this->network[this->network.size() - 1];
    this->network[this->network.size() - 1].inputs.emplace_back(buffer);
    vector<double> tmpWeightedSums(this->network[this->network.size() - 1].num_perceptrons);
    for (int i = 0; i < this->network[this->network.size() - 1].num_perceptrons; i++) {
        tmpWeightedSums[i] = this->network[this->network.size() - 1].bias[i];
        for (int j = 0; j < buffer.size(); j++) {
            tmpWeightedSums[i] += this->network[this->network.size() - 1].weights[j][i] * buffer[j];
        }
    }
    this->network[this->network.size() - 1].weightedSums.emplace_back(tmpWeightedSums);

    // Bleh this is ugly
    double max = *max_element(tmpWeightedSums.begin(), tmpWeightedSums.end());
    for (double & tmpWeightedSum : tmpWeightedSums) {
        tmpWeightedSum -= max;
    }

    double wholeSum = 0.0;
    for (double weightedSum: tmpWeightedSums) {
        wholeSum += exp(weightedSum);
    }

    size_t length = this->network[this->network.size() - 1].num_perceptrons;
    vector<double> output(length);
    for (int i = 0; i < this->network[this->network.size() - 1].num_perceptrons; i++) {
        output[i] = exp(tmpWeightedSums[i]) / wholeSum;
    }

    this->network[this->network.size() - 1].activatedWeightedSums.emplace_back(output);
    return output;
}

void NeuralNetwork::backPropagation(const vector<vector<double>> &_gradient) {
    // Check if the input is the same size
    vector<vector<double>> gradient = _gradient;
    /*
    for (int i = 0; i < gradient.size(); i++) {
        cout << "Gradient " << i << ": ";
        for (double j : gradient[i]) {
            cout << j << " ";
        }
        cout << endl;
    }
    */


    // Update bias of the last layer
    for (int i = 0; i < this->network[this->network.size() - 1].bias.size(); i++) {
        double sum = 0.0;
        for (int j = 0; j < gradient.size(); j++) {
            sum += gradient[j][i];
        }
        this->network[this->network.size() - 1].bias[i] -= this->network[this->network.size() - 1].learningRate * (sum / 100);
    }

    // Delta of weights from inputs from lower layer
    vector<vector<double>> deltaOfWeights(this->network[this->network.size() - 1].perceptrons_below,
                                          vector<double>(this->network[this->network.size() - 1].num_perceptrons, 0));
    for (int i = 0; i < deltaOfWeights.size(); i++) {
        for (int j = 0; j < deltaOfWeights[0].size(); j++) {
            double sum = 0.0;
            for (int k = 0; k < gradient.size(); k++) {
                double gradientValue = gradient[k][j];
                double inputValue = this->network[this->network.size() - 2].activatedWeightedSums[k][i];
                sum += gradientValue * inputValue;
            }
            deltaOfWeights[i][j] = sum / 100;
        }
    }

    // New gradient pass to the next layer
    vector<vector<double>> newGradient(gradient.size(),
                                       vector<double>(this->network[this->network.size() - 1].perceptrons_below, 0));
    for (int i = 0; i < newGradient.size(); i++) {
        for (int j = 0; j < newGradient[0].size(); j++) {
            double sum = 0.0;
            for (int k = 0; k < gradient[0].size(); k++) {
                sum += gradient[i][k] * this->network[this->network.size() - 1].weights[j][k];
            }
            newGradient[i][j] = sum / gradient[0].size();
        }
    }

    // Update weights
    for (int i = 0; i < this->network[this->network.size() - 1].weights.size(); i++) {
        for (int j = 0; j < this->network[this->network.size() - 1].weights[0].size(); j++) {
            this->network[this->network.size() - 1].weights[i][j] -= learningRate * deltaOfWeights[i][j];
        }
    }

    gradient = newGradient;
    for (int w = this->network.size() - 2; w >= 0; w--) {
        // Fix old gradient
        for (int i = 0; i < this->network[w].weightedSums.size(); i++) {
            for (int j = 0; j < this->network[w].weightedSums[0].size(); j++) {
                if (this->network[w].weightedSums[i][j] <= 0) {
                    gradient[i][j] = 0;
                }
            }
        }

        // Update bias of the last layer
        for (int i = 0; i < this->network[w].bias.size(); i++) {
            double sum = 0.0;
            for (int j = 0; j < gradient.size(); j++) {
                sum += gradient[j][i];
            }
            this->network[w].bias[i] -= this->network[w].learningRate * (sum / 100);
        }

        // Delta of weights from inputs from lower layer
        vector<vector<double>> deltaOfWeights(this->network[w].perceptrons_below,
                                              vector<double>(this->network[w].num_perceptrons, 0));
        for (int i = 0; i < deltaOfWeights.size(); i++) {
            for (int j = 0; j < deltaOfWeights[0].size(); j++) {
                double sum = 0.0;
                for (int k = 0; k < gradient.size(); k++) {
                    sum += gradient[k][j] * this->network[w].inputs[k][i];
                }
                deltaOfWeights[i][j] = (sum / 100);
            }
        }

        // New gradient pass to the next layer
        vector<vector<double>> newGradient(gradient.size(),
                                           vector<double>(this->network[w].perceptrons_below, 0));
        for (int i = 0; i < newGradient.size(); i++) {
            for (int j = 0; j < newGradient[0].size(); j++) {
                double sum = 0.0;
                for (int k = 0; k < gradient[0].size(); k++) {
                    sum += gradient[i][k] * this->network[w].weights[j][k];
                }
                newGradient[i][j] = (sum / gradient[0].size());
            }
        }

        // Update weights
        for (int i = 0; i < this->network[w].weights.size(); i++) {
            for (int j = 0; j < this->network[w].weights[0].size(); j++) {
                this->network[w].weights[i][j] -= learningRate * deltaOfWeights[i][j];
            }
        }

        gradient = newGradient;
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

int getIndexOfHighestValue(const vector<double>& input) {
    int index = 0;
    for (int i = 1; i < input.size(); i++) {
        if (input[i] > input[index]) {
            index = i;
        }
    }
    return index;
}

vector<double> vectorHighestValue(const vector<double>& input) {
    int category = getIndexOfHighestValue(input);
    vector<double> output(input.size(), 0);
    output[category] = 1;
    return output;
}

void NeuralNetwork::accuracy(const vector<vector<double>> &inputs, const vector<vector<double>> &targets) {
    if (inputs.size() != targets.size()) {
        throw invalid_argument("NeuralNetwork::accuracy -> The number of inputs and targets are not the same!");
    }
    int correct = 0;
    for (int i = 0; i < inputs.size(); ++i) {
        if (getIndexOfHighestValue(inputs[i]) == getIndexOfHighestValue(targets[i])) {
            cout << "Index of highest value:" << getIndexOfHighestValue(inputs[i]) << endl;
            correct++;
        }
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
            for (auto &layer: this->network) {
                layer.clearBeforeBatch();
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
    vector<double> error(targets[0].size());
    vector<vector<double>> gradient = {};
    for (int i = 0; i < inputs.size(); i++) {
        vector<double> prediction = feedForward(inputs[i]);
        for (int j = 0; j < prediction.size(); j++) {
            error[j] = targets[i][j] - prediction[j];
        }
        gradient.emplace_back(error);
    }
    backPropagation(gradient);
}

/*Odpad
 //    if (gradient.size() != this->inputLayerSize) {
//        throw invalid_argument("NeuralNetwork::backPropagation -> The input is not correct, not forwarding further");
//    }
    // Check if the target is the same size
    //if (targets.size() != this->outputLayerSize) {
    //    throw invalid_argument("NeuralNetwork::backPropagation -> Target is not correct, not forwarding further");
    //}

    // Feed forward
    // vector<double> ff = feedForward(inputs);

    // Cost derivative
//    vector<double> zL = network[network.size() - 1].getWeightedSums();
//    for (int i = 0; i < zL.size(); i++) {
//        zL[i] = network[network.size() - 1].activationFunction.derivative(zL[i]);
//    }
//    vector<double> cost = costDerivative(ff, targets);
//    for (int i = 0; i < cost.size(); i++) {
//        cost[i] = cost[i] * zL[i];
//    }

    // Cost from Cross-Entropy
    // vector<double> cost = costDerivative(ff, targets);

    vector<double> outputsFromLowerLayer = getOutputsFromLowerLayer(this->network.size() - 1);
    this->network[this->network.size() - 1].updateWeightsAndBiases(cost, outputsFromLowerLayer);

    vector<double> costUpperLayer = cost;

    for (int i = this->network.size() - 2; i > 0; i--) {
        // Calculate cost
        vector<double> activatedSums = {};
        for (int j = 0; j < this->network[i].weightedSums.size(); j++) {
            activatedSums.emplace_back(
                    this->network[i].activationFunction.derivative(this->network[i].weightedSums[j]));
        }

        vector<double> delta = {};
        for (int k = 0; k < this->network[i + 1].weights.size(); k++) {
            double num = 0;
            for (int j = 0; j < costUpperLayer.size(); j++) {
                num += costUpperLayer[j] * this->network[i + 1].weights[j][k];
            }
            delta.emplace_back(num);
        }

        costUpperLayer = {};
        for (int j = 0; j < delta.size(); j++) {
            costUpperLayer.emplace_back(delta[j] * activatedSums[j]);
        }

        // Update
        outputsFromLowerLayer = getOutputsFromLowerLayer(i);
        this->network[i].updateWeightsAndBiases(costUpperLayer, outputsFromLowerLayer);
    }

    // plus input layer

    // Backpropagation

    for (int i = this->network.size() - 2; i >= 0; i--) {
        vector<double> activatedSums = {};
        for (int j = 0; j < this->network[i].weightedSums.size(); j++) {
            activatedSums.emplace_back(
                    this->network[i].activationFunction.derivative(this->network[i].weightedSums[j]));
        }

        vector<double> delta = {};
        for (int k = 0; k < this->network[i + 1].weights.size(); k++) {
            double num = 0;
            for (int j = 0; j < cost.size(); j++) {
                num += cost[j] * this->network[i + 1].weights[j][k];
            }
            delta.emplace_back(num);
        }

        cost = {};
        for (int j = 0; j < delta.size(); j++) {
            cost.emplace_back(delta[j] * activatedSums[j]);
        }

        // Update weights and biases
        this->network[i].updateBias(cost);
        if (i == 0) {
            for (int j = 0; j < this->network[0].weights.size(); j++) {
                for (int k = 0; k < this->network[0].weights[0].size(); k++) {
                    this->network[0].weights[j][k] -= inputs[j] * this->learningRate * cost[k];
                }
            }
            return;
        }


        for (int j = 0; j < this->network[i].weights.size(); j++) {
            for (int k = 0; k < this->network[i].weights[0].size(); k++) {
                this->network[i].weights[j][k] -= outputsFromLowerLayer[j] * this->learningRate * cost[k];
            }
        }
    }
    */

