import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Random;

public class Util {
    static Random rd = new Random();

    public static void saveNetwork(String path, Network network) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(new File(path))));
            oos.writeObject(network);
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Network loadNetwork(String path) {
        try {
            ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(new File(path))));
            Network network = (Network) ois.readObject();
            ois.close();
            return network;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static double[] linspace(double start, double end, int total) {
        double[] doubles = new double[total];
        for (int i = 0; i < total; i++) {
            doubles[i] = start + i * (end - start) / (total - 1);
        }
        return doubles;
    }

    public static double[] random(double start, double end, int total) {
        double[] doubles = new double[total];
        for (int i = 0; i < total; i++) {
            doubles[i] = (end - start) * rd.nextDouble() + start;
        }
        return doubles;
    }

    public static double[] imgInfo(String src, String position, int offset) {
        BufferedImage bf = readImage(src);
        int[] rgbArray = convertImageToArray(bf, position, offset);
        return getImgInfo(rgbArray);
    }

    private static BufferedImage readImage(String imageFile) {
        File file = new File(imageFile);
        BufferedImage bf = null;
        try {
            bf = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bf;
    }

    private static int[] convertImageToArray(BufferedImage bf, String position, int offset) {
        int width = bf.getWidth();
        int height = bf.getHeight();
        int[] data = new int[width * height];
        switch (position) {
            case "center":
                bf.getRGB(0, 0, width, height, data, 0, width);
                break;
            case "left":
                bf.getRGB(offset, 0, width - offset, height, data, offset, width);
                break;
            case "right":
                bf.getRGB(0, 0, width - offset, height, data, 0, width);
                break;
            case "up":
                bf.getRGB(0, offset, width, height - offset, data, 0, width);
                break;
            case "down":
                bf.getRGB(0, 0, width, height - offset, data, 0, width);
                break;
        }
        return data;
    }

    private static double[] getImgInfo(int[] rgbArray) {
        double[] imgInfo = new double[rgbArray.length];
        for (int i = 0; i < rgbArray.length; i++) {
            if (rgbArray[i] == -1) {
                imgInfo[i] = 0;
            } else {
                imgInfo[i] = 1;
            }
        }
        return imgInfo;
    }
}
