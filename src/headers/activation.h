class Activation
{
public:
    static double identity(double sum);
    static double logistic(double sum);
    static double tanh(double sum);
    static double relu(double sum);

    static void parseActivationFuction(string activation, double (**activationFunction)(double sum));
};
