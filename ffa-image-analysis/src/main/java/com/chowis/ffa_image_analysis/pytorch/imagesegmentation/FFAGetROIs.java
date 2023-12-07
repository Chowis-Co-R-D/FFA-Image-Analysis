package com.chowis.ffa_image_analysis.pytorch.imagesegmentation;

import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;
import static org.opencv.imgproc.Imgproc.INTER_AREA;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.resize;

import android.content.Context;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.pytorch.IValue;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;

public class FFAGetROIs {

    //
    // ---------- Segment the Face Regional ROIs with FA detectron2.
    //
    public static void doAnalysis(Context context,
                                  String pplFrontImgPath,
                                  String pplLeftImgPath,
                                  String pplRightImgPath,
                                  Module d2Module,
                                  String frontFullFaceMaskOutputPath,
                                  String leftFullFaceMaskOutputPath,
                                  String rightFullFaceMaskOutputPath,
                                  String poresROIOutputPath,
                                  String oilinessFrontROIOutputPath,
                                  String radianceFrontROIOutputPath,
                                  String frontForeheadMaskOutputPath,
                                  String frontNoseMaskOutputPath,
                                  String frontCheekMaskOutputPath,
                                  String frontChinMaskOutputPath,
                                  String leftForeheadMaskOutputPath,
                                  String leftNoseMaskOutputPath,
                                  String leftCheekMaskOutputPath,
                                  String leftChinMaskOutputPath,
                                  String rightForeheadMaskOutputPath,
                                  String rightNoseMaskOutputPath,
                                  String rightCheekMaskOutputPath,
                                  String rightChinMaskOutputPath) {

        Mat frontOriginalPPL = new Mat();
        if (pplFrontImgPath != null) frontOriginalPPL = Imgcodecs.imread(pplFrontImgPath);

        Mat leftOriginalPPL = new Mat();
        if (pplLeftImgPath != null) leftOriginalPPL = Imgcodecs.imread(pplLeftImgPath);

        Mat rightOriginalPPL = new Mat();
        if (pplRightImgPath != null) rightOriginalPPL = Imgcodecs.imread(pplRightImgPath);

        final int originalHeight = frontOriginalPPL.rows();
        final int originalWidth = frontOriginalPPL.cols();
        final int channelsNum = frontOriginalPPL.channels();

        Mat frontChinMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat frontFullFaceMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat frontForeheadMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat frontCheekMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat frontNoseMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);

        Mat leftChinMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat leftFullFaceMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat leftForeheadMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat leftCheekMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat leftNoseMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);

        Mat rightChinMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat rightFullFaceMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat rightForeheadMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat rightCheekMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);
        Mat rightNoseMask = Mat.zeros(originalHeight, originalWidth, CV_8UC1);

        if (!frontOriginalPPL.empty() && !leftOriginalPPL.empty() && !rightOriginalPPL.empty()) {
            // ----- (1) ----- Load detectron2 optimized TorchScript model.
            for (int positionCount = 0; positionCount < 3; positionCount++) {
                // position 0: front
                // position 1: left
                // position 2: right
                //
                // ------ (2) ------ Prepare input tensor for detectron2 from opencv2 Mat image.
                Mat inputImg = new Mat();
                if (positionCount == 0) {
                    inputImg = frontOriginalPPL.clone();
                }
                if (positionCount == 1) {
                    inputImg = leftOriginalPPL.clone();
                }
                if (positionCount == 2) {
                    inputImg = rightOriginalPPL.clone();
                }
                final int inputWidth = 244;     // 782 mapped to 800....
                final int inputHeight = 299;       // 1038 mysteriously mapped to 1067


                if (originalHeight * originalWidth < inputHeight * inputWidth) {
                    resize(inputImg, inputImg, new Size(inputWidth, inputHeight), INTER_LINEAR);
                } else if (originalHeight * originalWidth > inputHeight * inputWidth) {
                    resize(inputImg, inputImg, new Size(inputWidth, inputHeight), INTER_AREA);
                }

                // Extract data from OpenCV image and create input tensor for detectron2.
                byte[] imgData = new byte[inputHeight * inputWidth * channelsNum];
                inputImg.get(0, 0, imgData);

                float[] permutedImgData = new float[inputHeight * inputWidth * channelsNum];

                int m = 0;
                for (int i = 0; i < channelsNum; i++) {
                    for (int j = 0; j < inputHeight; j++) {
                        for (int k = 0; k < inputWidth; k++) {
                            // Must use &0xFF to convert "uchar" to unsigned 8-bit bytes first, then to float
                            // otherwise we will get negative values.
                            permutedImgData[m] = imgData[(j * inputWidth + k) * channelsNum + i] & 0xFF;
                            m = m + 1;
                        }
                    }
                }
                Log.println(Log.VERBOSE, "created tensor data block is of length:", String.valueOf(permutedImgData.length));

                org.pytorch.Tensor tensorDetectron2 = org.pytorch.Tensor.fromBlob(permutedImgData, new long[]{channelsNum, inputHeight, inputWidth}, MemoryFormat.CONTIGUOUS);

                // Input tensor validation.
                long[] tensorSize = tensorDetectron2.shape();
                Log.println(Log.VERBOSE, "Tensor shape 0: ", String.valueOf(tensorSize[0]));
                Log.println(Log.VERBOSE, "Tensor shape 1: ", String.valueOf(tensorSize[1]));
                Log.println(Log.VERBOSE, "Tensor shape 2: ", String.valueOf(tensorSize[2]));

                IValue inputD2 = IValue.from(tensorDetectron2);
                Log.println(Log.VERBOSE, "Checking detectron2 input tensor status", String.valueOf(inputD2.isTensor()));
                org.pytorch.Tensor tensorDebug = inputD2.toTensor();
                final float[] tensorDebugData = tensorDebug.getDataAsFloatArray();
                Log.println(Log.VERBOSE, "debugging tensor data block size: ", String.valueOf(tensorDebugData.length));

                Log.d("ImageSegmentation", "-- SHU LI: Image tensor created from OPENCV");

                // ----- (3) ----- Inference and validate outputs.
                IValue d2output = d2Module.forward(IValue.from(tensorDetectron2));
                IValue[] d2outputTensors = d2output.toTuple();

                org.pytorch.Tensor bbox, pred_classes, pred_masks, d2scores;
                bbox = d2outputTensors[0].toTensor();
                pred_classes = d2outputTensors[1].toTensor();
                pred_masks = d2outputTensors[2].toTensor();
                d2scores = d2outputTensors[3].toTensor();

                Log.println(Log.VERBOSE, "Number of bounding boxes is:", String.valueOf(bbox.numel() / 4));
                Log.println(Log.VERBOSE, "Number of class instances is:", String.valueOf(pred_classes.numel()));
                Log.println(Log.VERBOSE, "Number of predicted masks is:", String.valueOf(pred_masks.numel() / (28 * 28)));
                Log.println(Log.VERBOSE, "Score count is:", String.valueOf(d2scores.numel()));

                final long[] classIDs = pred_classes.getDataAsLongArray();
                for (int i = 0; i < classIDs.length; i++) {
                    Log.println(Log.VERBOSE, "Prediced Class ID is: ", String.valueOf(classIDs[i]));
                }

                Log.d("ImageSegmentation", "-- SHU LI: detectron2 output obtained.");

                // ----- (4) ----- Process outputs and create mask images.
                int num_instances = (int) d2scores.numel();

                Mat foreheadMask = Mat.zeros(inputHeight, inputWidth, CV_8UC1);
                Mat noseMask = Mat.zeros(inputHeight, inputWidth, CV_8UC1);
                Mat cheekMask = Mat.zeros(inputHeight, inputWidth, CV_8UC1);
                Mat fullFaceROIMask = Mat.zeros(inputHeight, inputWidth, CV_8UC1);
                Mat chinMask = Mat.zeros(inputHeight, inputWidth, CV_8UC1);

                final float[] extractedMasks = pred_masks.getDataAsFloatArray();

                final float[] extractedBoxes = bbox.getDataAsFloatArray();
                for (int i = 0; i < num_instances; i++) {
                    // convert continuous box coordinates to discrete box coordinates.
                    // bbox size is 4.
                    int boxSize = 4;
                    int samples_w = 1 + (int) (extractedBoxes[i * boxSize + 2]) - (int) (extractedBoxes[i * boxSize + 0]);
                    int samples_h = 1 + (int) (extractedBoxes[i * boxSize + 3]) - (int) (extractedBoxes[i * boxSize + 1]);

                    // convert instance mask tensor to OpenCV Mat image.
                    // instance mask size is 28 * 28.
                    int inst_mask_w = 28;
                    int inst_mask_h = 28;
                    int maskSize = inst_mask_w * inst_mask_h;

                    float threshold = 0.5F;

                    Mat mask = Mat.zeros(inst_mask_h, inst_mask_w, CV_8UC1);

                    for (int j = 0; j < inst_mask_w; j++) {
                        for (int k = 0; k < inst_mask_h; k++) {
                            if (extractedMasks[i * maskSize + j * inst_mask_w + k] >= threshold) {
                                int pixValue = 255;
                                mask.put(j, k, pixValue);
                            }
                        }
                    }

                    resize(mask, mask, new Size(samples_w, samples_h), 0, 0, INTER_LINEAR);

                    // Target rectangle ROI coordinates inside the original image (or say final mask image for the instance).
                    int x_0 = Math.max((int) (extractedBoxes[i * boxSize + 0]), 0);   // x0.
                    int x_1 = Math.min((int) (extractedBoxes[i * boxSize + 2]) + 1, inputWidth);  // (x_1 - x_0) is width.
                    int y_0 = Math.max((int) (extractedBoxes[i * boxSize + 1]), 0);   // y0.
                    int y_1 = Math.min((int) (extractedBoxes[i * boxSize + 3]) + 1, inputHeight); // (y_1 - y_0) is height.

                    Mat resultImg = Mat.zeros(inputHeight, inputWidth, CV_8UC1);
                    for (int x = x_0; x < x_1; x++) {
                        for (int y = y_0; y < y_1; y++) {
                            if ((y - y_0) < mask.rows() && (x - x_0) < mask.cols()) {
                                // This IF condition is necessary for error control.
                                // In rare situations, y-y_0 (or x-x_0) is 1 unit greater/less than mask.rows() (or mask.cols).
                                // This is caused by conversion of float numbers to int.
                                int pixValue = (int) ((mask.get(y - y_0, x - x_0))[0]);
                                resultImg.put(y, x, pixValue);
                            }
                        }
                    }

                    if (classIDs[i] == 0)
                        org.opencv.core.Core.bitwise_or(chinMask, resultImg, chinMask);
                    if (classIDs[i] == 1)
                        org.opencv.core.Core.bitwise_or(fullFaceROIMask, resultImg, fullFaceROIMask);
                    if (classIDs[i] == 2)
                        org.opencv.core.Core.bitwise_or(foreheadMask, resultImg, foreheadMask);
                    if (classIDs[i] == 3)
                        org.opencv.core.Core.bitwise_or(cheekMask, resultImg, cheekMask);
                    if (classIDs[i] == 4)
                        org.opencv.core.Core.bitwise_or(noseMask, resultImg, noseMask);
                    if (classIDs[i] == 5)
                        org.opencv.core.Core.bitwise_or(cheekMask, resultImg, cheekMask);
                }

                if (originalHeight * originalWidth < inputHeight * inputWidth) {
                    resize(foreheadMask, foreheadMask, new Size(originalWidth, originalHeight), INTER_AREA);
                    resize(noseMask, noseMask, new Size(originalWidth, originalHeight), INTER_AREA);
                    resize(cheekMask, cheekMask, new Size(originalWidth, originalHeight), INTER_AREA);
                    resize(fullFaceROIMask, fullFaceROIMask, new Size(originalWidth, originalHeight), INTER_AREA);
                    resize(chinMask, chinMask, new Size(originalWidth, originalHeight), INTER_AREA);
                } else if (originalHeight * originalWidth > inputHeight * inputWidth) {
                    resize(foreheadMask, foreheadMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
                    resize(noseMask, noseMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
                    resize(cheekMask, cheekMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
                    resize(fullFaceROIMask, fullFaceROIMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
                    resize(chinMask, chinMask, new Size(originalWidth, originalHeight), INTER_LINEAR);
                }

                //MyUtil.saveMatToGallery(context, "dummy forehead", "mask containing foreheads detected", foreheadMask);
                //MyUtil.saveMatToGallery(context, "dummy nose", "mask containing detected nose", noseMask);
                //MyUtil.saveMatToGallery(context, "dummy cheek", "mask containing detected cheek", cheekMask);
                //MyUtil.saveMatToGallery(context, "dummy full face", "mask containing detected full face", fullFaceROIMask);
                //MyUtil.saveMatToGallery(context, "dummy chin", "mask containing detected chin", chinMask);

                if (positionCount == 0) {
                    // front
                    frontChinMask = chinMask.clone();
                    frontFullFaceMask = fullFaceROIMask.clone();
                    frontForeheadMask = foreheadMask.clone();
                    frontCheekMask = cheekMask.clone();
                    frontNoseMask = noseMask.clone();

                    chinMask.release();
                    fullFaceROIMask.release();
                    foreheadMask.release();
                    cheekMask.release();
                    noseMask.release();
                }
                if (positionCount == 1) {
                    // left
                    leftChinMask = chinMask.clone();
                    leftFullFaceMask = fullFaceROIMask.clone();
                    leftForeheadMask = foreheadMask.clone();
                    leftCheekMask = cheekMask.clone();
                    leftNoseMask = noseMask.clone();

                    chinMask.release();
                    fullFaceROIMask.release();
                    foreheadMask.release();
                    cheekMask.release();
                    noseMask.release();
                }
                if (positionCount == 2) {
                    // right
                    rightChinMask = chinMask.clone();
                    rightFullFaceMask = fullFaceROIMask.clone();
                    rightForeheadMask = foreheadMask.clone();
                    rightCheekMask = cheekMask.clone();
                    rightNoseMask = noseMask.clone();

                    chinMask.release();
                    fullFaceROIMask.release();
                    foreheadMask.release();
                    cheekMask.release();
                    noseMask.release();
                }
            }
        }

        // ------ (-1) ------ Save regional ROI mask images.
        //
        if (frontForeheadMaskOutputPath != null)
            imwrite(frontForeheadMaskOutputPath, frontForeheadMask);
        if (frontNoseMaskOutputPath != null) imwrite(frontNoseMaskOutputPath, frontNoseMask);
        if (frontCheekMaskOutputPath != null) imwrite(frontCheekMaskOutputPath, frontCheekMask);
        if (frontChinMaskOutputPath != null) imwrite(frontChinMaskOutputPath, frontChinMask);

        if (leftForeheadMaskOutputPath != null)
            imwrite(leftForeheadMaskOutputPath, leftForeheadMask);
        if (leftNoseMaskOutputPath != null) imwrite(leftNoseMaskOutputPath, leftNoseMask);
        if (leftCheekMaskOutputPath != null) imwrite(leftCheekMaskOutputPath, leftCheekMask);
        if (leftChinMaskOutputPath != null) imwrite(leftChinMaskOutputPath, leftChinMask);

        if (rightForeheadMaskOutputPath != null)
            imwrite(rightForeheadMaskOutputPath, rightForeheadMask);
        if (rightNoseMaskOutputPath != null) imwrite(rightNoseMaskOutputPath, rightNoseMask);
        if (rightCheekMaskOutputPath != null) imwrite(rightCheekMaskOutputPath, rightCheekMask);
        if (rightChinMaskOutputPath != null) imwrite(rightChinMaskOutputPath, rightChinMask);

        // ------ (0) ------ Begin preparation of ROI images for algorithms.
        //
        Imgproc.cvtColor(frontChinMask, frontChinMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(frontForeheadMask, frontForeheadMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(frontCheekMask, frontCheekMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(frontNoseMask, frontNoseMask, COLOR_GRAY2BGR);

        Imgproc.cvtColor(leftChinMask, leftChinMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(leftForeheadMask, leftForeheadMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(leftCheekMask, leftCheekMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(leftNoseMask, leftNoseMask, COLOR_GRAY2BGR);

        Imgproc.cvtColor(rightChinMask, rightChinMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(rightForeheadMask, rightForeheadMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(rightCheekMask, rightCheekMask, COLOR_GRAY2BGR);
        Imgproc.cvtColor(rightNoseMask, rightNoseMask, COLOR_GRAY2BGR);

        // ------ (1) ------ Save full face ROI mask images.
        //
        if (frontFullFaceMaskOutputPath != null)
            Imgcodecs.imwrite(frontFullFaceMaskOutputPath, frontFullFaceMask);
        if (leftFullFaceMaskOutputPath != null)
            Imgcodecs.imwrite(leftFullFaceMaskOutputPath, leftFullFaceMask);
        if (rightFullFaceMaskOutputPath != null)
            Imgcodecs.imwrite(rightFullFaceMaskOutputPath, rightFullFaceMask);

        // ------ (2) ------ Prepare and save ROI images for pores.
        //
        Mat poresROI = new Mat();

        Core.bitwise_or(frontNoseMask, frontCheekMask, poresROI);
        bitwise_and(poresROI, frontOriginalPPL, poresROI);

        if (poresROIOutputPath != null)
            Imgcodecs.imwrite(poresROIOutputPath, poresROI);

        //MyUtil.saveMatToGallery(context, "dummy front pores ROI", "front face pores ROI", poresROI);

        // ------ (3) ------ Prepare and save ROI image for oiliness.
        //
        Mat oilinessFrontROI = new Mat();

        Core.bitwise_or(frontChinMask, frontForeheadMask, oilinessFrontROI);
        Core.bitwise_or(oilinessFrontROI, frontCheekMask, oilinessFrontROI);
        Core.bitwise_or(oilinessFrontROI, frontNoseMask, oilinessFrontROI);
        bitwise_and(oilinessFrontROI, frontOriginalPPL, oilinessFrontROI);

        if (oilinessFrontROIOutputPath != null)
            Imgcodecs.imwrite(oilinessFrontROIOutputPath, oilinessFrontROI);

        //MyUtil.saveMatToGallery(context, "dummy oiliness ROI", "oiliness face ROI", oilinessFrontROI);

        // ----- (4) ----- Prepare and save ROI image for Radiance and Dullness.
        //
        Mat radianceFrontROI = new Mat();

        Core.bitwise_or(frontForeheadMask, frontCheekMask, radianceFrontROI);
        bitwise_and(radianceFrontROI, frontOriginalPPL, radianceFrontROI);

        if (radianceFrontROIOutputPath != null)
            Imgcodecs.imwrite(radianceFrontROIOutputPath, radianceFrontROI);

        //MyUtil.saveMatToGallery(context, "dummy radiance ROI", "radiance face ROI", radianceFrontROI);

        // Get anonymized images for saving to database.
        Mat anonymizedFrontImg = new Mat();
        Mat anonymizedLeftImg = new Mat();
        Mat anonymizedRightImg = new Mat();

        cvtColor(frontFullFaceMask, frontFullFaceMask, COLOR_GRAY2BGR);
        cvtColor(leftFullFaceMask, leftFullFaceMask, COLOR_GRAY2BGR);
        cvtColor(rightFullFaceMask, rightFullFaceMask, COLOR_GRAY2BGR);

        bitwise_and(frontFullFaceMask, frontOriginalPPL, anonymizedFrontImg);
        bitwise_and(leftFullFaceMask, leftOriginalPPL, anonymizedLeftImg);
        bitwise_and(rightFullFaceMask, rightOriginalPPL, anonymizedRightImg);
    }
}