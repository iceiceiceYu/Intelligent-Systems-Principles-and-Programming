public class Regression {
    public static void main(String[] args) {
        int[] structure = new int[]{1, 10, 1};
        Network regNN = new Network(structure, false, 0.03, 0.002);

        double[] trainSet = Util.random(-Math.PI, Math.PI, 1000);
        double[] testSet = Util.random(-Math.PI, Math.PI, 4000);
        double[] trainDesired = new double[trainSet.length];
        double[] testDesired = new double[testSet.length];

        for (int i = 0; i < trainDesired.length; i++) {
            trainDesired[i] = Math.sin(trainSet[i]);
        }

        for (int i = 0; i < testDesired.length; i++) {
            testDesired[i] = Math.sin(testSet[i]);
        }

        double startTime = System.currentTimeMillis();
        train(regNN, 10000, trainSet, trainDesired);
        double endTime = System.currentTimeMillis();
        test(regNN, testSet, testDesired);
        System.out.println("total run time: " + (endTime - startTime) + " ms");

        testSet = Util.random(-Math.PI, Math.PI, 8000);
        testDesired = new double[testSet.length];
        for (int i = 0; i < testDesired.length; i++) {
            testDesired[i] = Math.sin(testSet[i]);
        }
        test(regNN, testSet, testDesired);


        testSet = Util.random(-Math.PI, Math.PI, 10000);
        testDesired = new double[testSet.length];
        for (int i = 0; i < testDesired.length; i++) {
            testDesired[i] = Math.sin(testSet[i]);
        }
        test(regNN, testSet, testDesired);

        testSet = Util.random(-Math.PI, Math.PI, 20000);
        testDesired = new double[testSet.length];
        for (int i = 0; i < testDesired.length; i++) {
            testDesired[i] = Math.sin(testSet[i]);
        }
        test(regNN, testSet, testDesired);

        testSet = Util.random(-Math.PI, Math.PI, 60000);
        testDesired = new double[testSet.length];
        for (int i = 0; i < testDesired.length; i++) {
            testDesired[i] = Math.sin(testSet[i]);
        }
        test(regNN, testSet, testDesired);

        testSet = Util.random(-Math.PI, Math.PI, 100);
        testDesired = new double[testSet.length];
        for (int i = 0; i < testDesired.length; i++) {
            testDesired[i] = Math.sin(testSet[i]);
        }
        test(regNN, testSet, testDesired);

        testSet = Util.random(-Math.PI, Math.PI, 10);
        testDesired = new double[testSet.length];
        for (int i = 0; i < testDesired.length; i++) {
            testDesired[i] = Math.sin(testSet[i]);
        }
        test(regNN, testSet, testDesired);

        testSet = Util.random(-Math.PI, Math.PI, 500);
        testDesired = new double[testSet.length];
        for (int i = 0; i < testDesired.length; i++) {
            testDesired[i] = Math.sin(testSet[i]);
        }
        test(regNN, testSet, testDesired);
    }

    public static void train(Network network, int iteration, double[] input, double[] desired) {
        double error;
        double output;
        for (int i = 0; i < iteration; i++) {
            error = 0;
            for (int j = 0; j < input.length; j++) {
                network.forward(new double[]{input[j]});
                network.backward(new double[]{desired[j]});
                output = network.layers.get(network.size - 1).neurons[0].output;
                error += Math.abs(desired[j] - output);
            }
            System.out.println("training: epoch " + (i + 1) + ", average error is: " + (error / input.length));
        }
    }

    public static void test(Network network, double[] input, double[] desired) {
        double error = 0;
        double output;
        for (int i = 0; i < input.length; i++) {
            network.forward(new double[]{input[i]});
            output = network.layers.get(network.size - 1).neurons[0].output;
            error += Math.abs(desired[i] - output);
        }
        System.out.println("test set average error: " + (error / input.length));
    }
}
