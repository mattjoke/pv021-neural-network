#include <iostream>
#include <algorithm>
using namespace std;

class ActivationFunction
{
private:
    ActivationFunction(auto function)
    {
        this->function = []() {};
    }

public:
    static const ActivationFunction *identity;
};
