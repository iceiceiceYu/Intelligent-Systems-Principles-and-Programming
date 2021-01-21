import java.util.*;

public class Classification {
    static final int TOTAL_TYPE = 12;
    static final int TRAIN_SET_SIZE = 620;

    public static void main(String[] args) {
        int[] structure = {784, 64, 12};
        Network claNN = new Network(structure, true, 0.005, 0.002);

        Set<Integer> train = new HashSet<>();
//        Set<Integer> test = new HashSet<>();

        for (int i = 1; i <= TRAIN_SET_SIZE; i++) {
            train.add(i);
        }

        train(claNN, train);
        Util.saveNetwork("network/claNNInterview6", claNN);
    }

    public static void train(Network network, Set<Integer> train) {
        List<Integer> integers = new ArrayList<>();
        for (int i = 1; i <= TOTAL_TYPE; i++) {
            integers.add(i);
        }
        double[] input;
        double[] desired;
        double maxProbability;
        int predict;
        double total = 0;
        double correct = 0;
        Neuron[] neurons;
        String[] position = new String[]{"center", "left", "right", "up", "down"};

        for (int a = 0; a < 10; a++) {
            for (Integer j : train) {
                //Collections.shuffle(integers);
                for (int i : integers) {
                    for (int c = 1; c < 4; c++) {
                        for (int b = 0; b < 5; b++) {
                            //for (int i = 1; i <= TOTAL_TYPE; i++) {
                            maxProbability = 0;
                            predict = 0;
                            input = Util.imgInfo("train/" + i + "/" + j + ".bmp", position[b], c);
                            desired = new double[12];
                            desired[i - 1] = 1;

                            network.forward(input);

                            network.softmax();

                            network.backward(desired);

                            neurons = network.layers.get(network.size - 1).neurons;
                            for (int k = 0; k < TOTAL_TYPE; k++) {
                                if (neurons[k].output > maxProbability) {
                                    maxProbability = neurons[k].output;
                                    predict = k + 1;
                                }
                            }

                            if (predict == i) {
                                correct++;
                            }
                            total++;
                        }
                        System.out.println("training times " + total + ", current correctness: " + (correct / total));
                    }
                }
            }
        }
    }

    public static void test(Network network, Set<Integer> test) {
        List<Integer> integers = new ArrayList<>();
        for (int i = 1; i <= TOTAL_TYPE; i++) {
            integers.add(i);
        }
        double[] input;
        double[] desired;
        double maxProbability;
        int predict;
        double total = 0;
        double correct = 0;
        Neuron[] neurons;
        String[] position = new String[]{"center", "left", "right", "up", "down"};

        for (Integer j : test) {
            Collections.shuffle(integers);
            for (int i : integers) {
                for (int c = 1; c < 3; c++) {
                    for (int b = 0; b < 5; b++) {
                        maxProbability = 0;
                        predict = 0;
                        input = Util.imgInfo("train/" + i + "/" + j + ".bmp", position[b], c);
                        desired = new double[12];
                        desired[i - 1] = 1;

                        network.forward(input);
                        network.softmax();

                        neurons = network.layers.get(network.size - 1).neurons;
                        for (int k = 0; k < TOTAL_TYPE; k++) {
                            if (neurons[k].output > maxProbability) {
                                maxProbability = neurons[k].output;
                                predict = k + 1;
                            }
                        }
                        if (predict == i) {
                            correct++;
                        }
                        total++;
                    }
                }
            }
        }
        System.out.println("test case " + total + ", total correctness: " + (correct / total));
    }
}
