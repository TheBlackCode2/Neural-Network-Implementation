#include <iostream>
#include <cstdint>
#include <vector>
#include <cmath>
#include <random>


double GetRandom()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dist(-1.0f, 1.0f);

    return dist(gen);
}

// Sigmoid
double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double SigmoidDerivative(double s) { return s * (1.0 - s); }

// Tanh
double Tanhf(double x) { return std::tanh(x); }
double TanhfDerivative(double s) { return 1.0 - s * s; }


class NeuralNetwork
{
public:
    NeuralNetwork(int32_t input_size, int32_t hidden1_size, int32_t hidden2_size, int32_t output_size)
    : m_InputSize(input_size), m_Hidden1Size(hidden1_size), m_Hidden2Size(hidden2_size), m_OutputSize(output_size) 
    {
        Initialize();
    }

    std::vector<double> FeedForward(const std::vector<double>& input);
    void Train(const std::vector<double>& input, const std::vector<double>& target, double learn_rate);

private:
    int32_t m_InputSize, m_Hidden1Size, m_Hidden2Size, m_OutputSize;

    std::vector<std::vector<double>> m_InputToHidden1Weights, m_Hidden1ToHidden2Weights, m_Hidden2ToOutputWeights;
    std::vector<double> m_Hidden1Neuron, m_Hidden2Neuron;
    std::vector<double> m_Hidden1Bias, m_Hidden2Bias, m_OutputBias;

    void Initialize();

    inline double Activation(double x) { return Tanhf(x); }
    inline double ActivationDerivative(double s) { return TanhfDerivative(s); }
};

void NeuralNetwork::Initialize()
{
    m_InputToHidden1Weights.resize(m_InputSize, std::vector<double>(m_Hidden1Size, 0.0));
    m_Hidden1ToHidden2Weights.resize(m_Hidden1Size, std::vector<double>(m_Hidden2Size, 0.0));
    m_Hidden2ToOutputWeights.resize(m_Hidden2Size, std::vector<double>(m_OutputSize, 0.0));

    m_Hidden1Bias.resize(m_Hidden1Size, 0.0);
    m_Hidden2Bias.resize(m_Hidden2Size, 0.0);
    m_OutputBias.resize(m_OutputSize, 0.0);

    m_Hidden1Neuron.resize(m_Hidden1Size, 0.0);
    m_Hidden2Neuron.resize(m_Hidden2Size, 0.0);

    for (size_t i = 0; i < m_InputSize; i++)
        for (size_t j = 0; j < m_Hidden1Size; j++)
            m_InputToHidden1Weights[i][j] = GetRandom();

    for (size_t i = 0; i < m_Hidden1Size; i++)
        for (size_t j = 0; j < m_Hidden2Size; j++)
            m_Hidden1ToHidden2Weights[i][j] = GetRandom();

    for (size_t i = 0; i < m_Hidden2Size; i++)
        for (size_t j = 0; j < m_OutputSize; j++)
            m_Hidden2ToOutputWeights[i][j] = GetRandom();


    for (size_t i = 0; i < m_Hidden1Size; i++)
        m_Hidden1Bias[i] = GetRandom();

    for (size_t i = 0; i < m_Hidden2Size; i++)
        m_Hidden2Bias[i] = GetRandom();

    for (size_t i = 0; i < m_OutputSize; i++)
        m_OutputBias[i] = GetRandom();
}

std::vector<double> NeuralNetwork::FeedForward(const std::vector<double>& input)
{
    std::vector<double> output(m_OutputSize, 0.0);

    for (size_t i = 0; i < m_Hidden1Size; i++)
    {
        m_Hidden1Neuron[i] = 0.0;
        for (size_t j = 0; j < m_InputSize; j++)
            m_Hidden1Neuron[i] += input[j] * m_InputToHidden1Weights[j][i];
        m_Hidden1Neuron[i] += m_Hidden1Bias[i];
        m_Hidden1Neuron[i] = Activation(m_Hidden1Neuron[i]);
    }

    for (size_t i = 0; i < m_Hidden2Size; i++)
    {
        m_Hidden2Neuron[i] = 0.0;
        for (size_t j = 0; j < m_Hidden1Size; j++)
            m_Hidden2Neuron[i] += m_Hidden1Neuron[j] * m_Hidden1ToHidden2Weights[j][i];
        m_Hidden2Neuron[i] += m_Hidden2Bias[i];
        m_Hidden2Neuron[i] = Activation(m_Hidden2Neuron[i]);
    }

    for (size_t i = 0; i < m_OutputSize; i++)
    {
        output[i] = 0.0;
        for (size_t j = 0; j < m_Hidden2Size; j++)
            output[i] += m_Hidden2Neuron[j] * m_Hidden2ToOutputWeights[j][i];
        output[i] += m_OutputBias[i];
        output[i] = Activation(output[i]);
    }

    return output;
}

void NeuralNetwork::Train(const std::vector<double>& input, const std::vector<double>& target, double learn_rate)
{
    std::vector<double> output = FeedForward(input);

    // Output Errors & Deltas
    std::vector<double> outputErrors(m_OutputSize, 0.0);
    for (size_t i = 0; i < m_OutputSize; i++)
        outputErrors[i] = target[i] - output[i];

    std::vector<double> outputDeltas(m_OutputSize, 0.0);
    for (size_t i = 0; i < m_OutputSize; i++)
        outputDeltas[i] = 2 * outputErrors[i] * ActivationDerivative(output[i]);

    // Hidden2 Errors & Deltas
    std::vector<double> hidden2Errors(m_Hidden2Size, 0.0);
    for (size_t i = 0; i < m_Hidden2Size; i++)
        for (size_t j = 0; j < m_OutputSize; j++)
        hidden2Errors[i] += outputDeltas[j] * m_Hidden2ToOutputWeights[i][j];

    std::vector<double> hidden2Deltas(m_Hidden2Size, 0.0);
    for (size_t i = 0; i < m_Hidden2Size; i++)
        for (size_t j = 0; j < m_OutputSize; j++)
        hidden2Deltas[i] += hidden2Errors[j] * ActivationDerivative(m_Hidden2Neuron[i]);

    // Hidden1 Errors & Deltas
    std::vector<double> hidden1Errors(m_Hidden1Size, 0.0);
    for (size_t i = 0; i < m_Hidden1Size; i++)
        for (size_t j = 0; j < m_Hidden2Size; j++)
        hidden1Errors[i] += hidden2Deltas[j] * m_Hidden1ToHidden2Weights[i][j];

    std::vector<double> hidden1Deltas(m_Hidden1Size, 0.0);
    for (size_t i = 0; i < m_Hidden1Size; i++)
        for (size_t j = 0; j < m_Hidden2Size; j++)
        hidden1Deltas[i] += hidden1Errors[j] * ActivationDerivative(m_Hidden1Neuron[i]);

    // Update Weights & Bias
    for (size_t i = 0; i < m_InputSize; i++)
        for (size_t j = 0; j < m_Hidden1Size; j++)
            m_InputToHidden1Weights[i][j] += hidden1Deltas[j] * input[i] * learn_rate;

    for (size_t i = 0; i < m_Hidden1Size; i++)
        for (size_t j = 0; j < m_Hidden2Size; j++)
            m_Hidden1ToHidden2Weights[i][j] += hidden2Deltas[j] * m_Hidden1Neuron[i] * learn_rate;

    for (size_t i = 0; i < m_Hidden2Size; i++)
        for (size_t j = 0; j < m_OutputSize; j++)
            m_Hidden2ToOutputWeights[i][j] += outputDeltas[j] * m_Hidden2Neuron[i] * learn_rate;

    
    for (size_t i = 0; i < m_Hidden1Size; i++)
        m_Hidden1Bias[i] += hidden1Deltas[i] * learn_rate;

    for (size_t i = 0; i < m_Hidden2Size; i++)
        m_Hidden2Bias[i] += hidden2Deltas[i] * learn_rate;

    for (size_t i = 0; i < m_OutputSize; i++)
        m_OutputBias[i] += outputDeltas[i] * learn_rate;
}

int main()
{
    // XOR
    std::vector<std::vector<double>> inputs  = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<std::vector<double>> outputs = {    {0},    {1},    {1},    {0} };

    NeuralNetwork nn(2, 4, 4, 1);

    int32_t epochs = 5000;
    double learnRate = 0.1;

    for (size_t epoch = 0; epoch < epochs; epoch++)
        for (size_t i = 0; i < inputs.size(); i++)
            nn.Train(inputs[i], outputs[i], learnRate);

    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<double> output = nn.FeedForward(inputs[i]);
        std::cout << inputs[i][0] << ", " << inputs[i][1] << " => " << output[0] << "\n";
    }

    return 0;
}