import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Network implements Serializable {
    int size;
    List<Layer> layers;

    public Network(int[] structure, boolean classification, double weightLR, double biasLR) {
        this.size = structure.length;
        this.layers = new ArrayList<>();
        Layer backLayer = null;
        Layer layer;

        for (int i : structure) {
            layer = new Layer(i, backLayer, classification, weightLR, biasLR);
            if (backLayer != null) {
                backLayer.nextLayer = layer;
            }
            backLayer = layer;
            layers.add(layer);
        }
    }

    public void forward(double[] input) {
        Layer layer = layers.get(0);
        Neuron[] neurons = layer.neurons;
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].output = input[i]; // Input Layer
        }
        for (int i = 1; i < size; i++) {
            layers.get(i).forward();
        }
    }

    public void backward(double[] desired) {
        Layer layer = layers.get(size - 1);
        Neuron[] neurons = layer.neurons;
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].delta = desired[i] - neurons[i].output; // Output Layer
        }
        for (int i = layers.size() - 1; i > 0; i--) {
            layers.get(i).backward(); // Output Layer & Hidden Layer
        }
    }

    void softmax() {
        Layer layer = layers.get(size - 1);
        Neuron[] neurons = layer.neurons;
        double totalOutput = 0;
        for (Neuron neuron : neurons) {
            totalOutput += Math.exp(neuron.output);
        }
        for (Neuron neuron : neurons) {
            neuron.output = Math.exp(neuron.output) / totalOutput;
        }
    }
}
