#include <iostream>
#include <algorithm>
using namespace std;

class Perceptron
{
private:
    double *weights;
    int activationFunction(double sum)
    {
        return sum >= 0 ? 1 : 0;
    }

public:
    Perceptron(size_t size)
    {
        this->weights = new double[size]();
        for (size_t i = 0; i < size; i++)
        {
            this->weights[i] = 0;
        }
    }
    int feedForward(double *inputs)
    {
        for (double input : inputs)
        {
            /* code */
        }
        return activationFunction(1.1 + 1);
    }
};
