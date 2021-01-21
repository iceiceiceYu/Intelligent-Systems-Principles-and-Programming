import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class Interview {
    static final int TOTAL_TYPE = 12;
    static final int TEST_AMOUNT = 1800;

    public static void main(String[] args) {
        Network claNN = Util.loadNetwork("network/claNNInterview5");
        double[][] testSet = loadData();
        int[] prediction = new int[TEST_AMOUNT];
        int count = 0;
        for (double[] testData : testSet) {
            assert claNN != null;
            prediction[count++] = predict(claNN, testData);
        }
        output(prediction);
    }

    public static double[][] loadData() {
        double[][] testSet = new double[TEST_AMOUNT][784];
        int count = 0;
        for (int i = 1; i <= TEST_AMOUNT; i++) {
            testSet[count++] = Util.imgInfo("test/" + i + ".bmp", "center", 0);
        }
        return testSet;
    }

    public static int predict(Network network, double[] data) {
        double maxProbability = 0;
        int predict = 0;
        network.forward(data);
        network.softmax();
        Neuron[] neurons = network.layers.get(network.size - 1).neurons;
        for (int i = 0; i < TOTAL_TYPE; i++) {
            if (neurons[i].output > maxProbability) {
                maxProbability = neurons[i].output;
                predict = i + 1;
            }
        }
        return predict;
    }

    public static void output(int[] prediction) {
        try {
            PrintWriter writer = new PrintWriter(new File("pred.txt"));
            for (int i : prediction) {
                writer.println(i);
            }
            writer.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
