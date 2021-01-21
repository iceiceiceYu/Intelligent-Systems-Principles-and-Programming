import java.io.Serializable;
import java.util.Random;

public class Neuron implements Serializable {
    Random rd = new Random();
    double bias;
    double[] weights;
    double output;
    double delta;

    public Neuron() {
        this.weights = new double[0];
    }

    public Neuron(double bias, int backLayerQuantity, boolean classification) {
        this.bias = bias;
        this.weights = new double[backLayerQuantity];
        for (int i = 0; i < backLayerQuantity; i++) {
            if (classification) {
                weights[i] = rd.nextGaussian() / Math.sqrt(backLayerQuantity);
                //weights[i] = 0.02 * rd.nextDouble() - 0.01;
            } else {
                //weights[i] = rd.nextGaussian();
                weights[i] = rd.nextDouble();
            }
        }
    }

    static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    static double derivativeSigmoid(double x) {
        return x * (1 - x);
    }

    static double tanh(double x) {
        return (Math.exp(x) - Math.exp(-x) / Math.exp(x) + Math.exp(-x));
    }

    static double derivativeTanh(double x) {
        return 1 - x * x;
    }

    static double LeRU(double x) {
        return Math.max(0, x);
    }

    static double derivativeLeRU(double x) {
        return x > 0 ? 1 : 0;
    }
}
