#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

static const double LEARNING_RATE = 1;

class Perceptron
{
private:
    double *weights;
    size_t size;
    int activationFunction(double sum)
    {
        return sum >= 0 ? 1 : 0;
        // return 1 / (1 + exp(sum));
    }

public:
    Perceptron(int size)
    {
        this->size = size;
        this->weights = new double[this->size];
        for (size_t i = 0; i < this->size; i++)
        {
            this->weights[i] = i % 2 == 0 ? 1 : -1;
        }
    }
    ~Perceptron()
    {
        delete this->weights;
    }
    int feedForward(double *inputs)
    {
        double sum = 0;
        // there is assumption, that the inputs is at least this->size big
        for (size_t i = 0; i < this->size; i++)
        {
            sum += inputs[i] * this->weights[i];
        }

        return activationFunction(sum);
    }

    void train(double *inputs, int desired)
    {
        int guess = feedForward(inputs);
        int error = desired - guess;

        for (size_t i = 0; i < this->size; i++)
        {
            this->weights[i] += error * inputs[i] * LEARNING_RATE;
        }
    }

    void printWeigthVector()
    {
        string s = "Weight vector: [";
        for (size_t i = 0; i < this->size; i++)
        {
            s += to_string(this->weights[i]);
            if (i + 1 < this->size)
            {
                s += ",";
            }
        }
        s += "]\n";
        cout << s;
    }
};
