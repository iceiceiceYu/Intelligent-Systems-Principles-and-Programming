import java.io.Serializable;
import java.util.Random;

public class Layer implements Serializable {
    //static final double LAMBDA = 0.00001;
    //static final double lambdaLR = 0.005;
    Random rd = new Random();
    int quantity;
    Neuron[] neurons;
    Layer backLayer;
    Layer nextLayer;
    double weightLR;
    double biasLR;

    public Layer(int quantity, Layer backLayer, boolean classification, double weightLR, double biasLR) {
        this.quantity = quantity;
        this.neurons = new Neuron[quantity];
        this.backLayer = backLayer;
        if (backLayer != null) { // Hidden Layer
            int backLayerQuantity = backLayer.quantity;
            for (int i = 0; i < quantity; i++) {
                if (classification) {
                    this.neurons[i] = new Neuron(0, backLayerQuantity, true);
                    this.weightLR = weightLR;
                    this.biasLR = biasLR;
                } else {
                    this.neurons[i] = new Neuron(0.009 * rd.nextDouble() - 0.01, backLayerQuantity, false);
                    this.weightLR = weightLR;
                    this.biasLR = biasLR;
                }
            }
        } else { // Input Layer
            for (int i = 0; i < quantity; i++) {
                this.neurons[i] = new Neuron();
            }
        }
    }

    void forward() {
        Neuron[] backLayerNeurons = backLayer.neurons;
        for (Neuron neuron : neurons) {
            double sum = 0;
            for (int i = 0; i < neuron.weights.length; i++) {
                sum += neuron.weights[i] * backLayerNeurons[i].output;
            }
            sum += neuron.bias;
            if (nextLayer == null) { // Output Layer
                neuron.output = sum;
            } else { // Hidden Layer
                neuron.output = Neuron.sigmoid(sum);
                //neuron.output = Neuron.tanh(sum);
                //neuron.output = Neuron.LeRU(sum);
            }
        }
    }

    void backward() {
        Neuron[] backLayerNeurons = backLayer.neurons;
        Neuron[] nextLayerNeurons;
        if (nextLayer == null) { // Output Layer
            for (Neuron neuron : neurons) {
                double gradient = 1;
                gradient *= neuron.delta;
                for (int i = 0; i < neuron.weights.length; i++) {
                    //neuron.weights[i] *= lambdaLR * (1 - LAMBDA);
                    neuron.weights[i] += weightLR * gradient * backLayerNeurons[i].output;
                }
                neuron.bias += biasLR * gradient;
            }
        } else { // Hidden Layer
            nextLayerNeurons = nextLayer.neurons;
            for (int i = 0; i < neurons.length; i++) {
                double gradient = 0;
                for (Neuron nextLayerNeuron : nextLayerNeurons) {
                    gradient += nextLayerNeuron.delta * nextLayerNeuron.weights[i];
                }
                gradient *= Neuron.derivativeSigmoid(neurons[i].output);
                //gradient *= Neuron.derivativeTanh(neurons[i].output);
                //gradient *= Neuron.derivativeLeRU(neurons[i].output);
                neurons[i].delta = gradient;
                for (int j = 0; j < neurons[i].weights.length; j++) {
                    //neurons[i].weights[i] *= lambdaLR * (1 - LAMBDA);
                    neurons[i].weights[j] += weightLR * gradient * backLayerNeurons[j].output;
                }
                neurons[i].bias += biasLR * gradient;
            }
        }
    }
}
