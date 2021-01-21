import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Util {
    public static void main(String[] args) throws IOException {
        BufferedReader reader;
        PrintWriter writer;
        List<String> sentence = new ArrayList<>();
        List<String> tags = new ArrayList<>();
        String line;

        reader = new BufferedReader(new InputStreamReader(new FileInputStream("../example_dataset/input.utf8")));
        while ((line = reader.readLine()) != null) {
            sentence.add(line);
        }

        reader = new BufferedReader(new InputStreamReader(new FileInputStream("../example_dataset/gold.utf8")));
        while ((line = reader.readLine()) != null) {
            tags.add(line);
        }

        reader = new BufferedReader(new InputStreamReader(new FileInputStream("../dataset/dataset1/train.utf8")));
        StringBuilder sent = new StringBuilder();
        StringBuilder tag = new StringBuilder();
        while ((line = reader.readLine()) != null) {
            if (line.equals("")) {
                sentence.add(sent.toString());
                tags.add(tag.toString());
                sent.delete(0, sent.length());
                tag.delete(0, tag.length());
            } else {
                sent.append(line.split(" ")[0]);
                tag.append(line.split(" ")[1]);
            }
        }

        writer = new PrintWriter("../example_dataset/input.utf8");
        for (String input : sentence) {
            writer.println(input);
        }

        writer = new PrintWriter("../example_dataset/gold.utf8");
        for (String gold : tags) {
            writer.println(gold);
        }

        reader.close();
        writer.close();
    }
}
