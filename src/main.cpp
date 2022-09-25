#include <iostream>
#include "perceptron.cpp"
using namespace std;

int main()
{
    cout << "Hello! \n";
    Perceptron *p = new Perceptron(3);
    double point[3] = {50, -12, 1};
    int result = p->feedForward(point);
}
