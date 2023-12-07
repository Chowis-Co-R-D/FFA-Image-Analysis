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

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

public class FFAWrinklePigmentationSpots100 {

    //
    // ---------- Wrinkles & Spots Detection with FA detectron2.
    //
    public static String doAnalysis(Context context,
                                               Interpreter wrinkleSpotsModel,
                                               String frontOriginalInputPath,
                                               String leftOriginalInputPath,
                                               String rightOriginalInputPath,
                                               String frontFullFaceMaskInputPath,
                                               String leftFullFaceMaskInputPath,
                                               String rightFullFaceMaskInputPath,
                                               String frontWrinkleResultOutputPath,
                                               String leftWrinkleResultOutputPath,
                                               String rightWrinkleResultOutputPath,
                                               String frontWrinkleMaskOutputPath,
                                               String leftWrinkleMaskOutputPath,
                                               String rightWrinkleMaskOutputPath,
                                               String frontSpotsMaskOutputPath,
                                               String leftSpotsMaskOutputPath,
                                               String rightSpotsMaskOutputPath,
                                               boolean pigmentationSpotsSideImageEnabled) {

        Mat frontFullFaceMask = Imgcodecs.imread(frontFullFaceMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat leftFullFaceMask = Imgcodecs.imread(leftFullFaceMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat rightFullFaceMask = Imgcodecs.imread(rightFullFaceMaskInputPath, Imgcodecs.IMREAD_GRAYSCALE);

        Mat frontEyeWrinkleMask = new Mat();
        Mat frontForeheadWrinkleMask = new Mat();
        Mat frontLionWrinkleMask = new Mat();
        Mat frontSmileWrinkleMask = new Mat();

        Mat leftEyeWrinkleMask = new Mat();
        Mat leftForeheadWrinkleMask = new Mat();
        Mat leftLionWrinkleMask = new Mat();
        Mat leftSmileWrinkleMask = new Mat();

        Mat rightEyeWrinkleMask = new Mat();
        Mat rightForeheadWrinkleMask = new Mat();
        Mat rightLionWrinkleMask = new Mat();
        Mat rightSmileWrinkleMask = new Mat();

        Mat frontFullSpotsMask = new Mat(), leftFullSpotsMask = new Mat(), rightFullSpotsMask = new Mat();

        for (int positionCount = 0; positionCount < 3; positionCount++) {
            // ------------------------------------------ 1. AI inference ------------------------------------------
            //
            // Allocate input and output tensors.
            wrinkleSpotsModel.allocateTensors();

            // Prepare input tensor from OpenCV image.
            int[] inputShape = wrinkleSpotsModel.getInputTensor(0).shape(); // num, height, width, channel.
            int input_height = inputShape[1];
            int input_width = inputShape[2];

            // position 0: front
            // position 1: left
            // position 2: right
            //
            // ------ (2) ------ Prepare input tensor.
            Mat originalImg = new Mat();
            if (positionCount == 0) {
                originalImg = imread(frontOriginalInputPath);
            }
            if (positionCount == 1) {
                originalImg = imread(leftOriginalInputPath);
            }
            if (positionCount == 2) {
                originalImg = imread(rightOriginalInputPath);
            }

            // Resize image to our CFA detectron2 input size.
            final int originalHeight = originalImg.rows();
            final int originalWidth = originalImg.cols();

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
            int numClasses = 12;
            float[][][][] output = new float[1][input_height][input_width][numClasses];

            long timestamp2Start = System.currentTimeMillis();
            wrinkleSpotsModel.run(input, output);
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

            Mat eyeWrinkleMask = Mat.zeros(input_height, input_width, CV_8UC1);
            Mat foreheadWrinkleMask = Mat.zeros(input_height, input_width, CV_8UC1);
            Mat lionWrinkleMask = Mat.zeros(input_height, input_width, CV_8UC1);
            Mat smileWrinkleMask = Mat.zeros(input_height, input_width, CV_8UC1);
            Mat spotsMask = Mat.zeros(input_height, input_width, CV_8UC1);

            int backgroundClassID = 0;
            int acneSpotsClassID = 1;
            int acneClassID = 2;
            int spotsClassID = 3;
            int darkCircleClassID = 4;
            int eyeWrinkleClassID = 5;
            int foreheadWrinkleClassID = 6;
            int lionWrinkleClassID = 7;
            int molesClassID = 8;
            int pockmarkClassID = 9;
            int scarClassID = 10;
            int smileLineClassID = 11;
            for (int y = 0; y < input_height; y++) {
                for (int x = 0; x < input_width; x++) {
                    if(output[0][y][x][eyeWrinkleClassID] > 0.8) eyeWrinkleMask.put(y, x, 255);
                    if(output[0][y][x][foreheadWrinkleClassID] > 0.8) foreheadWrinkleMask.put(y, x, 255);
                    if(output[0][y][x][lionWrinkleClassID] > 0.8) lionWrinkleMask.put(y, x, 255);
                    if(output[0][y][x][smileLineClassID] > 0.8) smileWrinkleMask.put(y, x, 255);

                    if(output[0][y][x][acneSpotsClassID] > 0.8) spotsMask.put(y, x, 255);
                    if(output[0][y][x][acneClassID] > 0.8) spotsMask.put(y, x, 255);
                    if(output[0][y][x][spotsClassID] > 0.8) spotsMask.put(y, x, 255);
                    if(output[0][y][x][pockmarkClassID] > 0.8) spotsMask.put(y, x, 255);
                    if(output[0][y][x][scarClassID] > 0.8) spotsMask.put(y, x, 255);
                }
            }

            long timestamp3End = System.currentTimeMillis();
            Log.println(Log.VERBOSE, "Timestamp pores output: ", String.valueOf(timestamp3End - timestamp3Start));

            if (originalHeight * originalWidth < input_height * input_width) {
                resize(eyeWrinkleMask, eyeWrinkleMask, new Size(originalWidth, originalHeight), INTER_AREA);
                resize(foreheadWrinkleMask, foreheadWrinkleMask, new Size(originalWidth, originalHeight), INTER_AREA);
                resize(lionWrinkleMask, lionWrinkleMask, new Size(originalWidth, originalHeight), INTER_AREA);
                resize(smileWrinkleMask, smileWrinkleMask, new Size(originalWidth, originalHeight), INTER_AREA);
                resize(spotsMask, spotsMask, new Size(originalWidth, originalHeight), INTER_AREA);
            } else if (originalHeight * originalWidth > input_height * input_width) {
                resize(eyeWrinkleMask, eyeWrinkleMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
                resize(foreheadWrinkleMask, foreheadWrinkleMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
                resize(lionWrinkleMask, lionWrinkleMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
                resize(smileWrinkleMask, smileWrinkleMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
                resize(spotsMask, spotsMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
            }

            if (positionCount == 0) {
                // front
                frontEyeWrinkleMask = eyeWrinkleMask.clone();
                frontForeheadWrinkleMask = foreheadWrinkleMask.clone();
                frontLionWrinkleMask = lionWrinkleMask.clone();
                frontSmileWrinkleMask = smileWrinkleMask.clone();
                frontFullSpotsMask = spotsMask.clone();

                //MyUtil.saveMatToGallery(context, "front eye wrinkle", "mask containing detected eye wrinkle", frontEyeWrinkleMask);
                //MyUtil.saveMatToGallery(context, "front forehead wrinkle", "mask containing detected forehead wrinkle", frontForeheadWrinkleMask);
                //MyUtil.saveMatToGallery(context, "front lion wrinkle", "mask containing detected lion wrinkle", frontLionWrinkleMask);
                //MyUtil.saveMatToGallery(context, "front smile line", "mask containing detected smile line", frontSmileWrinkleMask);
                //MyUtil.saveMatToGallery(context, "front spots", "mask containing detected spots", frontFullSpotsMask);

                eyeWrinkleMask.release();
                foreheadWrinkleMask.release();
                lionWrinkleMask.release();
                smileWrinkleMask.release();
                spotsMask.release();
            }
            if (positionCount == 1) {
                // left
                leftEyeWrinkleMask = eyeWrinkleMask.clone();
                leftForeheadWrinkleMask = foreheadWrinkleMask.clone();
                leftLionWrinkleMask = lionWrinkleMask.clone();
                leftSmileWrinkleMask = smileWrinkleMask.clone();
                leftFullSpotsMask = spotsMask.clone();

                //MyUtil.saveMatToGallery(context, "left eye wrinkle", "mask containing detected eye wrinkle", leftEyeWrinkleMask);
                //MyUtil.saveMatToGallery(context, "left forehead wrinkle", "mask containing detected forehead wrinkle", leftForeheadWrinkleMask);
                //MyUtil.saveMatToGallery(context, "left lion wrinkle", "mask containing detected lion wrinkle", leftLionWrinkleMask);
                //MyUtil.saveMatToGallery(context, "left smile line", "mask containing detected smile line", leftSmileWrinkleMask);
                //MyUtil.saveMatToGallery(context, "front spots", "mask containing detected spots", leftFullSpotsMask);

                eyeWrinkleMask.release();
                foreheadWrinkleMask.release();
                lionWrinkleMask.release();
                smileWrinkleMask.release();
                spotsMask.release();
            }
            if (positionCount == 2) {
                // right
                rightEyeWrinkleMask = eyeWrinkleMask.clone();
                rightForeheadWrinkleMask = foreheadWrinkleMask.clone();
                rightLionWrinkleMask = lionWrinkleMask.clone();
                rightSmileWrinkleMask = smileWrinkleMask.clone();
                rightFullSpotsMask = spotsMask.clone();

                //MyUtil.saveMatToGallery(context, "right eye wrinkle", "mask containing detected eye wrinkle", rightEyeWrinkleMask);
                //MyUtil.saveMatToGallery(context, "right forehead wrinkle", "mask containing detected forehead wrinkle", rightForeheadWrinkleMask);
                //MyUtil.saveMatToGallery(context, "right lion wrinkle", "mask containing detected lion wrinkle", rightLionWrinkleMask);
                //MyUtil.saveMatToGallery(context, "right smile line", "mask containing detected smile line", rightSmileWrinkleMask);
                //MyUtil.saveMatToGallery(context, "front spots", "mask containing detected spots", rightFullSpotsMask);

                eyeWrinkleMask.release();
                foreheadWrinkleMask.release();
                lionWrinkleMask.release();
                smileWrinkleMask.release();
                spotsMask.release();
            }
        }
        // Save full face spots detection mask images for later spots indexing and scoring.
        if(!frontSpotsMaskOutputPath.equals("")) Imgcodecs.imwrite(frontSpotsMaskOutputPath, frontFullSpotsMask);
        if(pigmentationSpotsSideImageEnabled){
            if(!leftSpotsMaskOutputPath.equals("")) Imgcodecs.imwrite(leftSpotsMaskOutputPath, leftFullSpotsMask);
            if(!rightSpotsMaskOutputPath.equals(""))    Imgcodecs.imwrite(rightSpotsMaskOutputPath, rightFullSpotsMask);
        }

        // Wrinkle masks.
        Mat frontWrinkleFinalMask = new Mat();
        Mat leftWrinkleFinalMask = new Mat();
        Mat rightWrinkleFinalMask = new Mat();

        Core.bitwise_or(frontEyeWrinkleMask, frontForeheadWrinkleMask, frontWrinkleFinalMask);
        Core.bitwise_or(frontLionWrinkleMask, frontWrinkleFinalMask, frontWrinkleFinalMask);
        Core.bitwise_or(frontSmileWrinkleMask, frontWrinkleFinalMask, frontWrinkleFinalMask);

        Core.bitwise_or(leftEyeWrinkleMask, leftForeheadWrinkleMask, leftWrinkleFinalMask);
        Core.bitwise_or(leftLionWrinkleMask, leftWrinkleFinalMask, leftWrinkleFinalMask);
        Core.bitwise_or(leftSmileWrinkleMask, leftWrinkleFinalMask, leftWrinkleFinalMask);

        Core.bitwise_or(rightEyeWrinkleMask, rightForeheadWrinkleMask, rightWrinkleFinalMask);
        Core.bitwise_or(rightLionWrinkleMask, rightWrinkleFinalMask, rightWrinkleFinalMask);
        Core.bitwise_or(rightSmileWrinkleMask, rightWrinkleFinalMask, rightWrinkleFinalMask);

        //MyUtil.saveMatToGallery(context, "front full face wrinkle", "mask containing detected front face wrinkle", frontWrinkleFinalMask);
        //MyUtil.saveMatToGallery(context, "left full face wrinkle", "mask containing detected left face wrinkle", leftWrinkleFinalMask);
        //MyUtil.saveMatToGallery(context, "right full face wrinkle", "mask containing detected right face wrinkle", rightWrinkleFinalMask);

        // ----------------------------------- 2. AI Full Face ROI ------------------------------
        // Used for later indexing.
        Mat frontRoiMask = frontFullFaceMask;
        Mat leftRoiMask = leftFullFaceMask;
        Mat rightRoiMask = rightFullFaceMask;

        // ------------------------------------------ 3. Indexing ------------------------------------------
        //
        // Wrinkle scoring.
        double frontEyeScore = 0, frontForeheadScore = 0, frontLionScore = 0, frontSmileScore = 0;
        double leftEyeScore = 0, leftForeheadScore = 0, leftLionScore = 0, leftSmileScore = 0;
        double rightEyeScore= 0, rightForeheadScore = 0, rightLionScore = 0, rightSmileScore = 0;

        double frontEyeRaw = 0, frontForeheadRaw = 0, frontLionRaw = 0, frontSmileRaw = 0;
        double leftEyeRaw = 0, leftForeheadRaw = 0, leftLionRaw = 0, leftSmileRaw = 0;
        double rightEyeRaw = 0, rightForeheadRaw = 0, rightLionRaw = 0, rightSmileRaw = 0;

        double frontTotalRaw = 0, leftTotalRaw = 0, rightTotalRaw = 0;
        double frontTotalScore = 0, leftTotalScore = 0, rightTotalScore = 0;
        double allImagesWrinkleScore = 0;

        int frontFullFaceCount, leftFullFaceCount, rightFullFaceCount;
        frontFullFaceCount = countNonZero(frontRoiMask);
        leftFullFaceCount = countNonZero(leftRoiMask);
        rightFullFaceCount = countNonZero(rightRoiMask);

        if (frontFullFaceCount > 0) {
            frontEyeRaw = (double) 1000 * Core.countNonZero(frontEyeWrinkleMask) / frontFullFaceCount;
            frontForeheadRaw = (double) 1000 * Core.countNonZero(frontForeheadWrinkleMask) / frontFullFaceCount;
            frontLionRaw = (double) 1000 * Core.countNonZero(frontLionWrinkleMask) / frontFullFaceCount;
            frontSmileRaw = (double) 1000 * Core.countNonZero(frontSmileWrinkleMask) / frontFullFaceCount;
        }

        if (leftFullFaceCount > 0) {
            leftEyeRaw = (double) 1000 * Core.countNonZero(leftEyeWrinkleMask) / leftFullFaceCount;
            leftForeheadRaw = (double) 1000 * Core.countNonZero(leftForeheadWrinkleMask) / leftFullFaceCount;
            leftLionRaw = (double) 1000 * Core.countNonZero(leftLionWrinkleMask) / leftFullFaceCount;
            leftSmileRaw = (double) 1000 * Core.countNonZero(leftSmileWrinkleMask) / leftFullFaceCount;
        }

        if (rightFullFaceCount > 0) {
            rightEyeRaw = (double) 1000 * Core.countNonZero(rightEyeWrinkleMask) / rightFullFaceCount;
            rightForeheadRaw = (double) 1000 * Core.countNonZero(rightForeheadWrinkleMask) / rightFullFaceCount;
            rightLionRaw = (double) 1000 * Core.countNonZero(rightLionWrinkleMask) / rightFullFaceCount;
            rightSmileRaw = (double) 1000 * Core.countNonZero(rightSmileWrinkleMask) / rightFullFaceCount;
        }

        frontEyeScore = getCFAWrinkleLevel(frontEyeRaw, "eye");
        frontForeheadScore = getCFAWrinkleLevel(frontForeheadRaw, "forehead");
        frontLionScore = getCFAWrinkleLevel(frontLionRaw, "lion");
        frontSmileScore = getCFAWrinkleLevel(frontSmileRaw, "smile");

        leftEyeScore = getCFAWrinkleLevel(leftEyeRaw, "eye");
        leftForeheadScore = getCFAWrinkleLevel(leftForeheadRaw, "forehead");
        leftLionScore = getCFAWrinkleLevel(leftLionRaw, "lion");
        leftSmileScore = getCFAWrinkleLevel(leftSmileRaw, "smile");

        rightEyeScore = getCFAWrinkleLevel(rightEyeRaw, "eye");
        rightForeheadScore = getCFAWrinkleLevel(rightForeheadRaw, "forehead");
        rightLionScore = getCFAWrinkleLevel(rightLionRaw, "lion");
        rightSmileScore = getCFAWrinkleLevel(rightSmileRaw, "smile");

        if (frontFullFaceCount > 0) {
            frontTotalRaw = (double) 1000 * Core.countNonZero(frontWrinkleFinalMask) / frontFullFaceCount;
            leftTotalRaw = (double) 1000 * Core.countNonZero(leftWrinkleFinalMask) / leftFullFaceCount;
            rightTotalRaw = (double) 1000 * Core.countNonZero(rightWrinkleFinalMask) / rightFullFaceCount;
        }

        frontTotalScore = getCFAWrinkleLevel(frontTotalRaw, "frontTotal");
        leftTotalScore = getCFAWrinkleLevel(leftTotalRaw, "sideTotal");
        rightTotalScore = getCFAWrinkleLevel(rightTotalRaw, "sideTotal");

        allImagesWrinkleScore = Math.round(0.5 * frontTotalScore + 0.25 * leftTotalScore + 0.25 * rightTotalScore);

        // ------------------------------------------ 4. Prepare Output ------------------------------------------
        //
        // ----- (a) ----- Save wrinkle mask images first.
        Imgproc.cvtColor(frontWrinkleFinalMask, frontWrinkleFinalMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(leftWrinkleFinalMask, leftWrinkleFinalMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(rightWrinkleFinalMask, rightWrinkleFinalMask, COLOR_GRAY2BGR);

        Imgcodecs.imwrite(frontWrinkleMaskOutputPath, frontWrinkleFinalMask);
        Imgcodecs.imwrite(leftWrinkleMaskOutputPath, leftWrinkleFinalMask);
        Imgcodecs.imwrite(rightWrinkleMaskOutputPath, rightWrinkleFinalMask);

        // ----- (b) ------ Prepare input paths for mask images.
        String frontWrinkleMaskInputPath = frontWrinkleMaskOutputPath;
        String leftWrinkleMaskInputPath = leftWrinkleMaskOutputPath;
        String rightWrinkleMaskInputPath = rightWrinkleMaskOutputPath;

        int maskB = 0;
        int maskG = 145;
        int maskR = 255;
        int contourB = -1;
        int contourG = -1;
        int contourR = -1;
        double alpha = 0.55;

        com.chowis.jniimagepro.FFA.JNIFFAImageProCW myFFAImgProc = new JNIFFAImageProCW();
        double saveFrontRes = myFFAImgProc.FFAGetAnalyzedImgJni(frontOriginalInputPath, frontWrinkleMaskInputPath, frontWrinkleResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
        double saveLeftRes = myFFAImgProc.FFAGetAnalyzedImgJni(leftOriginalInputPath, leftWrinkleMaskInputPath, leftWrinkleResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);
        double saveRightRes = myFFAImgProc.FFAGetAnalyzedImgJni(rightOriginalInputPath, rightWrinkleMaskInputPath, rightWrinkleResultOutputPath, alpha, maskB, maskG, maskR, contourB, contourG, contourR);

        // ---------- Prepare returned values.
        //
        String returnString = allImagesWrinkleScore + "_" + frontTotalScore + "_" + leftTotalScore + "_" + rightTotalScore + "_" + frontTotalRaw + "_" + leftTotalRaw + "_" + rightTotalRaw + "_" + frontEyeScore + "_" + frontForeheadScore + "_" + frontLionScore + "_" + frontSmileScore + "_" + leftEyeScore + "_" + leftForeheadScore + "_" + leftLionScore + "_" + leftSmileScore + "_" + rightEyeScore + "_" + rightForeheadScore + "_" + rightLionScore + "_" + rightSmileScore + "_" + frontEyeRaw + "_" + frontForeheadRaw + "_" + frontLionRaw + "_" + frontSmileRaw + "_" + leftEyeRaw + "_" + leftForeheadRaw + "_" + leftLionRaw + "_" + leftSmileRaw + "_" + rightEyeRaw + "_" + rightForeheadRaw + "_" + rightLionRaw + "_" + rightSmileRaw;

        System.out.println("Returned String for Wrinkles and Spots is" + returnString);

        return returnString;
    }

    private static double getCFAWrinkleLevel(double pureValue, String position) {
        double[] dbNormData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int nMin = 9;
        int nMax = 9;
        int index = 9;

        if (position == "frontTotal") {
            dbNormData[0] = 0;
            dbNormData[1] = 0.92;
            dbNormData[2] = 1.50;
            dbNormData[3] = 2.00;
            dbNormData[4] = 2.66;
            dbNormData[5] = 3.37;
            dbNormData[6] = 4.30;
            dbNormData[7] = 5.15;
            dbNormData[8] = 6.45;
            dbNormData[9] = 9.92;
            dbNormData[10] = 22.09;
        } else if (position == "eye") {
            dbNormData[0] = 0.0;
            dbNormData[1] = 0.30;
            dbNormData[2] = 0.48;
            dbNormData[3] = 0.64;
            dbNormData[4] = 0.83;
            dbNormData[5] = 0.97;
            dbNormData[6] = 1.17;
            dbNormData[7] = 1.51;
            dbNormData[8] = 1.90;
            dbNormData[9] = 2.51;
            dbNormData[10] = 3.90;
        } else if (position == "forehead") {
            dbNormData[0] = 0.0;
            dbNormData[1] = 0.57;
            dbNormData[2] = 0.99;
            dbNormData[3] = 1.38;
            dbNormData[4] = 1.84;
            dbNormData[5] = 2.51;
            dbNormData[6] = 3.07;
            dbNormData[7] = 4.16;
            dbNormData[8] = 6.29;
            dbNormData[9] = 8.21;
            dbNormData[10] = 12.67;
        } else if (position == "lion") {
            dbNormData[0] = 0.0;
            dbNormData[1] = 0.53;
            dbNormData[2] = 0.66;
            dbNormData[3] = 0.82;
            dbNormData[4] = 1.00;
            dbNormData[5] = 1.14;
            dbNormData[6] = 1.29;
            dbNormData[7] = 1.51;
            dbNormData[8] = 1.82;
            dbNormData[9] = 2.23;
            dbNormData[10] = 2.99;
        } else if (position ==  "smile") {
            dbNormData[0] = 0.0;
            dbNormData[1] = 1.03;
            dbNormData[2] = 1.32;
            dbNormData[3] = 1.69;
            dbNormData[4] = 2.12;
            dbNormData[5] = 2.48;
            dbNormData[6] = 2.87;
            dbNormData[7] = 3.47;
            dbNormData[8] = 4.27;
            dbNormData[9] = 5.51;
            dbNormData[10] = 9.39;
        }
        else if(position == "sideTotal") {
            // for left and right
            dbNormData[0] = 0;
            dbNormData[1] = 0.23;
            dbNormData[2] = 0.56;
            dbNormData[3] = 0.78;
            dbNormData[4] = 1.03;
            dbNormData[5] = 1.24;
            dbNormData[6] = 1.41;
            dbNormData[7] = 1.90;
            dbNormData[8] = 2.35;
            dbNormData[9] = 4.22;
            dbNormData[10] = 13.90;
        }

        double nReturnValue;
        for (int i = 0; i < 10; i++) {
            if (pureValue >= dbNormData[i] && pureValue < dbNormData[i + 1]) {
                index = i;
                break;
            }
        }

        if (pureValue >= dbNormData[10]) {
            nReturnValue = 99;
        } else if (pureValue <= dbNormData[0]) {
            nReturnValue = 0;
        } else {
            nMin = index * 10;
            nMax = (index + 1) * 10;
            double dbMin = dbNormData[index];
            double dbMax = dbNormData[index + 1];
            nReturnValue = (int) (nMin + (nMax - nMin) * (pureValue - dbMin) / (dbMax - dbMin));
            if (nReturnValue > 99) {
                nReturnValue = 99;
            }
        }

        return Math.round(nReturnValue);
    }
}
