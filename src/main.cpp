#include "matrix.h"
#include "layer.h"
#include "perceptron.cpp"
#include "neural_network.cpp"
#include "image_holder.h"

using namespace std;

int main() {
    /*
    auto nn = NeuralNetwork(2, 2, {3});

    vector<vector<double>> dataset = {{0, 0},
                                      {0, 1},
                                      {1, 0},
                                      {1, 1}};
    // First column is 1, second column is 0
    vector<vector<double>> targets = {{0, 1}, // 0
                                      {1, 0}, // 1
                                      {1, 0}, // 1
                                      {0, 1}}; // 0

    // nn.printData();
    for (int i = 0; i < 10000; ++i) {
        // First output is 1 second is 0
        nn.trainBatch(dataset, targets);

        if (i % 1000 == 100) {
            cout << "Epoch: " << i << endl;
            auto predictions = nn.predict(dataset);
            nn.accuracy(predictions, targets);
        }
    }

    auto predictions = nn.predict(dataset);
    nn.accuracy(predictions, targets);
    cout << "-------------------" << endl;
    */
    string images_path = R"(../data/fashion_mnist_train_vectors.csv)";
    string labels_path = R"(../data/fashion_mnist_train_labels.csv)";
    cout << "Loading data..." << endl;
    auto ih = new ImageHolder(images_path, labels_path, 1000);
    cout << "Data loaded: " << ih->get_num_images() << " images" << endl;

    vector<vector<double>> images = ih->get_images(0, 100);
    vector<vector<double>> labels = ih->get_labels(0, 100);

    auto nn = NeuralNetwork(784, 10, {128, 64, 32});

    cout << "Training..." << endl;
    nn.train(images, labels);
    cout << "Training done" << endl;
    auto predictions = nn.predict(images);
    nn.accuracy(predictions, labels);
    return 0;
    /*


    auto *m = new Matrix(2, 3);
    m->add(+1);
    // matrix->printMatrix();

    auto *n = new Matrix(1, 2);

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
     */
}
