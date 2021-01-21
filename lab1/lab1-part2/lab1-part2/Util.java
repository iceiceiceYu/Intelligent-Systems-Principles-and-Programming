import java.io.*;

public class Util {
    final static int TRAIN_SIZE = 450;

    public static void main(String[] args) throws IOException {
//         String source = "/Users/yuzhexuan/pythonProject/lab1-part2/total_train";
//         String train = "/Users/yuzhexuan/pythonProject/lab1-part2/train";
//         String test = "/Users/yuzhexuan/pythonProject/lab1-part2/test";
//         for (int i = 1; i <= 12; i++) {
//             String sourcePath = source + "/" + i;
//             String trainPath = train + "/" + i;
//             String testPath = test + "/" + i;
//
//             for (int j = 1; j <= TRAIN_SIZE; j++) {
//                 copyFile(new File(sourcePath + "/" + j + ".bmp"), trainPath);
//             }
//
//             for (int j = TRAIN_SIZE + 1; j <= 620; j++) {
//                 copyFile(new File(sourcePath + "/" + j + ".bmp"), testPath);
//             }
//             System.out.println(i);
//         }
        //想命名的原文件的路径
        for (int i = 1; i <= 1800; i++) {

            File file = new File("interview_test/1/"+i+".bmp");

            file.renameTo(new File("interview_test/1/"+to4(i)+".bmp"));

        }
    }

    public static String to4(int i){
        if(i>=1 && i<=9){
            return "000"+i;
        }else if (i>=10 && i<=99) {
            return "00" + i;
        }else if(i>=100 && i<=999){
            return "0"+i;
        }else {
            return ""+i;
        }
    }

    private static void copyFile(File source, String dest) throws IOException {
        File destFile = new File(dest);
        if (!destFile.exists()) {
            destFile.mkdirs();
        }

        if (source.isDirectory()) {
            File dFile = new File(dest + "/" + source.getName());
            dFile.mkdirs();
            File[] files = source.listFiles();
            if (files.length != 0) {
                for (File value : files) {
                    copyFile(value, dFile.getPath());
                }
            }
        } else if (source.isFile()) {
            BufferedInputStream bis = new BufferedInputStream(new FileInputStream(source));

            File fFile = new File(dest + "/" + source.getName());

            if (!fFile.exists()) {
                fFile.createNewFile();
            }

            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(fFile));
            byte[] b = new byte[1024];
            int len;
            while ((len = bis.read(b)) != -1) {
                bos.write(b, 0, len);
            }

            bos.close();
            bis.close();
        }
    }
}