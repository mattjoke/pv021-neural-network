#include <iostream>
#include "matrix.h"
#include "layer.h"
#include "perceptron.cpp"
#include "neural_network.cpp"
#include "image_holder.h"

using namespace std;

int main() {
    auto nn = NeuralNetwork(2, 1, {});
    for (int i = 0; i < 100; ++i) {
        nn.train({1, 0}, {1});
        nn.train({1, 1}, {0});
        nn.train({0, 0}, {0});
        nn.train({0, 1}, {1});
    }

    cout << "Test" << endl;
    nn.predict({1, 0});

    return 0;
    while (1) {

        nn.feedForward({0, 0}).printMatrix();
    }



    auto *m = new Matrix(2, 3);
    m->add(+1);
    // matrix->printMatrix();

    auto *n = new Matrix(1, 2);

    n->randomise();
    n->printMatrix();

    n->transpose();
    // n->printMatrix();

    n->transpose();
    // n->printMatrix();
    // n->multiply(*matrix);
    // n->printMatrix();
    cout << "input\n";
    n->printMatrix();
    auto *l = new Layer(2, 3);
    cout << "weights\n";
    l->getWeights().printMatrix();
    cout << "bias\n";
    l->getBias().printMatrix();
    *n = l->feedForward(*n);
    n->printMatrix();

    // Oto
    m->printMatrix();
    n->printMatrix();
    // n->multiply(*matrix);
    // n->printMatrix();
    m->multiply(*n);
    m->printMatrix();
    n->printMatrix();

    cout << "-----" << endl;

    string images_path = R"(../data/fashion_mnist_test_vectors.csv)";
    string labels_path = R"(../data/fashion_mnist_test_labels.csv)";
    auto ih = new ImageHolder(images_path, labels_path);
    cout << ih->get_num_images() << endl;


    auto f = ih->get_image_as_matrix(0);


    auto *first = new Layer(784, 128);
    cout << "bias:" << endl;
    first->bias.printMatrix();


    auto *second = new Layer(128, 10);
    cout << "bias:" << endl;
    second->bias.printMatrix();

    second->feedForward(first->feedForward(f)).printMatrix();

}
