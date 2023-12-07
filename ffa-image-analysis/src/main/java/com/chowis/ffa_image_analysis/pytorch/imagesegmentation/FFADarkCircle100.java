package com.chowis.ffa_image_analysis.pytorch.imagesegmentation;

import static org.opencv.core.Core.countNonZero;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.resize;

import android.content.Context;
import android.util.Log;


import com.chowis.jniimagepro.FFA.JNIFFAImageProCW;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.util.ArrayList;
import java.util.List;

public class FFADarkCircle100 {


    //
    // ---------- Dark Circle Detection with AI.
    //
    public static String doAnalysis(Context context,
                                    Interpreter darkCircleModel,
                                    String frontOriginalInputPath,
                                    String frontFullFaceMaskInputPath,
                                    String darkCircleResultOutputPath,
                                    String darkCircleMaskOutputPath) {

        Mat frontFullFaceMask = imread(frontFullFaceMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);

        // ------------------------------------------ 1. AI inference ------------------------------------------
        //
        // Allocate input and output tensors.
        darkCircleModel.allocateTensors();

        // Prepare input tensor from OpenCV image.
        int[] inputShape = darkCircleModel.getInputTensor(0).shape(); // num, height, width, channel.
        int input_height = inputShape[1];
        int input_width = inputShape[2];

        Mat originalImg = imread(frontOriginalInputPath);

        int originalHeight = originalImg.height();
        int originalWidth = originalImg.width();

        cvtColor(originalImg, originalImg, COLOR_BGR2RGB);

        Mat inputImg = new Mat();

        if (originalHeight * originalWidth < input_height * input_width) {
            resize(originalImg, inputImg, new Size(input_width, input_height), INTER_LINEAR);
        } else if (originalHeight * originalWidth > input_height * input_width) {
            resize(originalImg, inputImg, new Size(input_width, input_height), INTER_AREA);
        }

        inputImg.convertTo(inputImg, CV_32F, ( 1 / 127.5), -1);

        long timestamp1Start = System.currentTimeMillis();
        float[][][][] input = new float[1][input_height][input_width][3];
        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < input_width; x++) {
                double[] pixel = inputImg.get(y, x);
                input[0][y][x][0] = (float)pixel[0];
                input[0][y][x][1] = (float)pixel[1];
                input[0][y][x][2] = (float)pixel[2];
            }
        }
        long timestamp1End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp input: ", String.valueOf(timestamp1End - timestamp1Start));

        //System.out.println("First normalized pixel value: " + input[0][0][0][0]);

        // Inference.
        int numClasses = 2;
        float[][][][] output = new float[1][input_height][input_width][numClasses];

        long timestamp2Start = System.currentTimeMillis();
        darkCircleModel.run(input, output);
        long timestamp2End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp inference: ", String.valueOf(timestamp2End - timestamp2Start));

        //int[] outputShape = darkCircleModel.getOutputTensor(0).shape();
        //System.out.println("output dim0: " + outputShape[0]);
        //System.out.println("output dim1: " + outputShape[1]);
        //System.out.println("output dim2: " + outputShape[2]);
        //System.out.println("output dim3: " + outputShape[3]);

        // Post processing.
        //
        // Get our wrinkle mask

        long timestamp3Start = System.currentTimeMillis();

        Mat darkCircleMask = Mat.zeros(new Size(input_width, input_height), CV_8UC1);
        int darkCircleClassID = 1;
        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < input_width; x++) {
                if(output[0][y][x][darkCircleClassID] > 0.8) darkCircleMask.put(y, x, 255);
            }
        }

        long timestamp3End = System.currentTimeMillis();
        Log.println(Log.VERBOSE, "Timestamp pores output: ", String.valueOf(timestamp3End - timestamp3Start));

        resize(darkCircleMask, darkCircleMask, originalImg.size());

        if (originalHeight * originalWidth < input_height * input_width) {
            resize(darkCircleMask, darkCircleMask, new Size(originalWidth, originalHeight), INTER_AREA);
        } else if (originalHeight * originalWidth > input_height * input_width) {
            resize(darkCircleMask, darkCircleMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
        }

        // front
        Mat frontDarkCircleMask = darkCircleMask.clone();
        //MyUtil.saveMatToGallery(context, "dark circle", "mask containing detected dark circle", frontDarkCircleMask);
        darkCircleMask.release();

        // ----------------------------------- 2. AI Full Face ROI ------------------------------
        // Used for later indexing.
        Mat frontRoiMask = frontFullFaceMask;

        // ------------------------------------------ 3. Indexing ------------------------------------------
        //
        int frontFullFaceCount = 0;
        frontFullFaceCount = countNonZero(frontRoiMask);

        // dark circle scoring
        double darkCircleRaw = 0, darkCircleScore = 0;
        if (frontFullFaceCount > 0) {
            darkCircleRaw = (double) 1000 * countNonZero(frontDarkCircleMask) / frontFullFaceCount;
        } else {
            darkCircleRaw = 0;
        }

        float[] dbNormData = {0.0f, 18.10f, 23.88f, 26.96f, 29.90f, 32.01f, 34.53f, 36.79f, 39.88f, 44.76f, 65.97f};

        int nMin = 9;
        int nMax = 9;
        int index = 9;

        for (int i = 0; i < 10; i++) {
            if (darkCircleRaw >= dbNormData[i] && darkCircleRaw < dbNormData[i + 1]) {
                index = i;
                break;
            }
        }

        if (darkCircleRaw >= dbNormData[10]) {
            darkCircleScore = 99;
        } else if (darkCircleRaw <= dbNormData[0]) {
            darkCircleScore = 0;
        } else {
            nMin = index * 10;
            nMax = (index + 1) * 10;
            float dbMin = dbNormData[index];
            float dbMax = dbNormData[index + 1];
            darkCircleScore = (double)(nMin + (nMax - nMin) * (darkCircleRaw - dbMin) / (dbMax - dbMin));
            if (darkCircleScore > 99) {
                darkCircleScore = 99;
            }
        }

        // ------------------------------------------ 4. Prepare Output ------------------------------------------
        //
        // Save dark cricle mask image first.
        Imgcodecs.imwrite(darkCircleMaskOutputPath, frontDarkCircleMask);

        // Dark Circle result image (analyzed image).
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(frontDarkCircleMask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        cvtColor(frontDarkCircleMask, frontDarkCircleMask, COLOR_GRAY2BGR);

        Scalar color = new Scalar(0, 255, 0);
        Imgproc.drawContours(frontDarkCircleMask, contours, -1, color, 2, Imgproc.LINE_AA, hierarchy, 0, new Point());

        int maskB = 245;
        int maskG = 52;
        int maskR = 245;
        int contourB = 230;
        int contourG = 0;
        int contourR = 165;
        double alpha = 0.6;

        String darkCircleMaskInputPath = darkCircleMaskOutputPath;

        JNIFFAImageProCW myFFAImgProc = new JNIFFAImageProCW();
        double saveDCres = myFFAImgProc.FFAGetAnalyzedImgJni(frontOriginalInputPath, darkCircleMaskInputPath, darkCircleResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);

        // ---------- Prepare returned values.
        //
        String returnString = darkCircleScore + "_" + darkCircleRaw;

        System.out.println("Returned String Dark Circles is" + returnString);

        return returnString;
    }
}
