package com.chowis.ffa_image_analysis.pytorch.imagesegmentation;


import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;
import static org.opencv.imgproc.Imgproc.cvtColor;

import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;


import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class MyUtil {
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public static String newFolderPath(Context context, String parentFolder, String folderName) throws IOException {
        File folder = new File(Environment.getExternalStorageDirectory() + parentFolder + folderName); // Create a File object with the folder path
        if (!folder.exists()) { // Check if the folder already exists
            boolean success = folder.mkdirs(); // Create the folder if it doesn't exist
            if (success) {
                System.out.println("Folder created successfully.");
            } else {
                System.out.println("Failed to create folder.");
            }
        } else {
            System.out.println("Folder already exists.");
        }

        return folder.getAbsolutePath();
    }

    public static String newFilePath(Context context, String parentFolder, String folderName, String fileName) throws IOException {
        File folder = new File(Environment.getExternalStorageDirectory() + parentFolder + folderName); // Create a File object with the folder path

        if (!folder.exists()) { // Check if the folder already exists
            boolean success = folder.mkdirs(); // Create the folder if it doesn't exist
            if (success) {
                System.out.println("Folder created successfully.");
            } else {
                System.out.println("Failed to create folder.");
            }
        } else {
            System.out.println("Folder already exists.");
        }

        String folderPath = folder.getAbsolutePath();

        File newFile = new File(folderPath, fileName);

        String filePath = newFile.getAbsolutePath();

        return filePath;
    }

    public static String createLocalBatchID() {
        Date currentDate = new Date();
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd");
        String formattedDate = dateFormat.format(currentDate);

        Calendar calendar = Calendar.getInstance();
        SimpleDateFormat timeFormat = new SimpleDateFormat("HHmm");
        String formattedTime = timeFormat.format(calendar.getTime());

        String localBatchID = formattedDate + formattedTime;
        System.out.println("Generated new batch ID: " + localBatchID);

        return localBatchID;
    }

    public static void saveMatToGallery(Context context, String imageName, String imageDescription, Mat inputMatImg) {
        if(inputMatImg.channels() == 3) cvtColor(inputMatImg, inputMatImg, COLOR_BGR2RGB);
        Bitmap bitmapImage = Bitmap.createBitmap(inputMatImg.cols(), inputMatImg.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(inputMatImg, bitmapImage);

        String title = imageName;
        String description = imageDescription;

        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE, title);
        values.put(MediaStore.Images.Media.DISPLAY_NAME, title);
        values.put(MediaStore.Images.Media.DESCRIPTION, description);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");

        Uri uri = context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        OutputStream outputStream = null;
        try {
            outputStream = context.getContentResolver().openOutputStream(uri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        bitmapImage.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
        try {
            outputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Mat getAnalyzedImg(Mat originalImg, Mat maskImg, double alpha, int maskB, int maskG, int maskR, int contourB, int contourG, int contourR) {
        long startTime = System.currentTimeMillis() / 1000;
        int originRows = originalImg.rows();
        int originCols = originalImg.cols();

        Log.println(Log.VERBOSE, "original image data type is: ", String.valueOf(originalImg.type()));

        double beta = 1 - alpha;
        for(int y = 0; y < originRows; y++){
            for (int x = 0; x < originCols; x++) {
                double[] maskImgPixel = maskImg.get(y, x);
                double[] originImgPixel = originalImg.get(y, x);
                if (maskImgPixel[0] == 255 && maskImgPixel[1] == 255 && maskImgPixel[2] == 255) {
                    double bv = alpha * originImgPixel[0] + beta * maskB;
                    double gv = alpha * originImgPixel[1] + beta * maskG;
                    double rv = alpha * originImgPixel[2] + beta * maskR;
                    byte[] newPixValue = {(byte)bv, (byte)gv, (byte)rv};

                    originalImg.put(y, x, newPixValue);
                }
                else if (maskImgPixel[0] == 0 && maskImgPixel[1] == 255 && maskImgPixel[2] == 0) {
                    double bv = alpha * originImgPixel[0] + beta * contourB;
                    double gv = alpha * originImgPixel[1] + beta * contourG;
                    double rv = alpha * originImgPixel[2] + beta * contourR;
                    byte[] newPixValue = {(byte)bv, (byte)gv, (byte)rv};

                    originalImg.put(y, x, newPixValue);
                }
            }
        }

        long endTime = System.currentTimeMillis() / 1000;
        Log.println(Log.VERBOSE, "Time to prepare one analyzed image is: ", String.valueOf(endTime - startTime));

        return originalImg;
    }
}
