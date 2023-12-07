package com.chowis.ffa_image_analysis;

import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import com.chowis.ffa_image_analysis.pytorch.imagesegmentation.FFADarkCircle100;
import com.chowis.ffa_image_analysis.pytorch.imagesegmentation.FFAGetROIs;
import com.chowis.ffa_image_analysis.pytorch.imagesegmentation.FFAWrinklePigmentationSpots100;
import com.chowis.ffa_image_analysis.pytorch.imagesegmentation.MyUtil;
import com.chowis.jniimagepro.FFA.JNIFFAImageProCW;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import com.google.mediapipe.imagesegmenter.SelfieSegmentation;

import org.opencv.core.Mat;
import org.pytorch.Module;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity implements Runnable {
    private String TAG = "MAIN_ACTIVITY";

    static {
        if (!NativeLoader.isInitialized()) {
            NativeLoader.init(new SystemDelegate());
        }
        NativeLoader.loadLibrary("torchvision_ops");
    }

    private Button mButtonSegment;
    private Module roiAIModule = null;  // pytorch AI model
    private String wrinkleSpotsModelPath = null;    // path
    private String darkCircleModelPath = null;  // path
    private Interpreter wrinkleSpotsTFModel = null;     // tensorflow AI model
    private Interpreter darkCircleTFModel = null;       // tensorflow AI model

    private String localBatchID = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        org.opencv.android.OpenCVLoader.initDebug();

        super.onCreate(savedInstanceState);

        // Load all AI models on creating.
        try {
            wrinkleSpotsModelPath = MyUtil.assetFilePath(getApplicationContext(), "ffa_unet_wrinkles_spots_v2.tflite");
            darkCircleModelPath = MyUtil.assetFilePath(getApplicationContext(), "ffa_darkcircles_v2.tflite");

            roiAIModule = org.pytorch.LiteModuleLoader.load(MyUtil.assetFilePath(getApplicationContext(), "FFA_roi_d2go.ptl"));
            Log.d(TAG, "onCreate: Load AI module");
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }

        File wrinkleSpotsModelFile = new File(wrinkleSpotsModelPath);
        File darkCircleModelFile = new File(darkCircleModelPath);

        wrinkleSpotsTFModel = new Interpreter(wrinkleSpotsModelFile);
        darkCircleTFModel = new Interpreter(darkCircleModelFile);
    }

    @Override
    public void run() {
        long startTime = System.currentTimeMillis() / 1000;

        // -------------- Generate local batch ID using current system date and time.
        localBatchID = MyUtil.createLocalBatchID();

        // -------------- Create folder for FFA results.
        String parentFolder = "/Documents/";
        String appFolderName = "DermoBellaSkin_FFA";
        String appFolder = null;
        try {
            appFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, appFolderName);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error creating APP folder", e);
        }

        // -------------- Create new folder for new BATCH in "Documents".
        parentFolder = "/Documents/" + appFolderName + "/";
        String newBatchFolderName = localBatchID;
        String newBatchFolder = null;
        try {
            newBatchFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, newBatchFolderName);
            System.out.println("Created new batch folder: " + newBatchFolder);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error creating new batch folder", e);
        }

        parentFolder = parentFolder + newBatchFolderName + "/";

        // ------------------------------ (0) Prepare Input Paths of Original Images ------------------------------
        //
        String frontPPLOriginalAssetPath = null;
        String leftPPLOriginalAssetPath = null;
        String rightPPLOriginalAssetPath = null;

        try {
            // input : original
            frontPPLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "ppl front.jpg");
            leftPPLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "ppl left.jpg");
            rightPPLOriginalAssetPath = MyUtil.assetFilePath(getApplicationContext(), "ppl right.jpg");
            Log.d(TAG, "run: input : original");
        } catch(IOException e) {
            e.printStackTrace();
        }

        // Create file paths for original images.
        String originalFrontPPLFolder = "original front ppl";
        String originalLeftPPLFolder = "original left ppl";
        String originalRightPPLFolder = "original right ppl";

        String originalFrontPPLFilename = "ppl front.jpg";
        String originalLeftPPLFilename = "ppl left.jpg";
        String originalRightPPLFilename = "ppl right.jpg";

        String originalFrontPPLPath = null;
        String originalLeftPPLPath = null;
        String originalRightPPLPath = null;

        try{
            originalFrontPPLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalFrontPPLFolder, originalFrontPPLFilename);
            originalLeftPPLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalLeftPPLFolder, originalLeftPPLFilename);
            originalRightPPLPath = MyUtil.newFilePath(getApplicationContext(), parentFolder, originalRightPPLFolder, originalRightPPLFilename);
            Log.d(TAG, "run: originalFrontPPLPath");
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating anonymized image output folders", e);
        }

        // Read files from assets, then save to their respective file paths.
        Mat frontPPLImg = imread(frontPPLOriginalAssetPath);
        Mat leftPPLImg = imread(leftPPLOriginalAssetPath);
        Mat rightPPLImg = imread(rightPPLOriginalAssetPath);

        imwrite(originalFrontPPLPath, frontPPLImg);
        imwrite(originalLeftPPLPath, leftPPLImg);
        imwrite(originalRightPPLPath, rightPPLImg);

        File frontPPL = new File(originalFrontPPLPath);
        File leftPPL = new File(originalLeftPPLPath);
        File rightPPL = new File(originalRightPPLPath);

        // ------------------------------ (1) Face Anonymization and Get Paths of Anonymized Images ------------------------------
        //
        // Paths for anonymized images.
        String anonymizedFrontImgFolderName = "anonymized front image";
        String anonymizedFrontImgFolder = null;

        String anonymizedLeftImgFolderName = "anonymized left image";
        String anonymizedLeftImgFolder = null;

        String anonymizedRightImgFolderName = "anonymized right image";
        String anonymizedRightImgFolder = null;

        try{
            anonymizedFrontImgFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, anonymizedFrontImgFolderName);
            anonymizedLeftImgFolder = MyUtil.newFolderPath(getApplication(), parentFolder, anonymizedLeftImgFolderName);
            anonymizedRightImgFolder = MyUtil.newFolderPath(getApplication(), parentFolder, anonymizedRightImgFolderName);
            Log.d(TAG, "run: anonymizedFrontImgFolder");
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating anonymized image output folders", e);
        }

        File anonymizedFrontImg = new File(anonymizedFrontImgFolder, "anonymized front face.jpg");
        File anonymizedLeftImg = new File(anonymizedLeftImgFolder, "anonymized left face.jpg");
        File anonymizedRightImg = new File(anonymizedRightImgFolder, "anonymized right face.jpg");

        String anonymizedFrontImgOutputPath = anonymizedFrontImg.getAbsolutePath();
        String anonymizedLeftImgOutputPath = anonymizedLeftImg.getAbsolutePath();
        String anonymizedRightImgOutputPath = anonymizedRightImg.getAbsolutePath();

        // Execute! Face anonymization.
        SelfieSegmentation segSelfie = new SelfieSegmentation(MainActivity.this);

        // running in coroutine already.
        segSelfie.runSegmentationOnImage(
                frontPPL,
                leftPPL,
                rightPPL,
                originalFrontPPLPath, anonymizedFrontImgOutputPath,
                originalLeftPPLPath, anonymizedLeftImgOutputPath,
                originalRightPPLPath, anonymizedRightImgOutputPath);

        long anonymizationTime = System.currentTimeMillis() / 1000;
        Log.d(TAG, "run: End of 1 step");
        Log.println(Log.VERBOSE, "Time for anonymization: ", String.valueOf(anonymizationTime - startTime));

        // ------------------------------ (2) FFA Get Face ROIs ------------------------------
        //
        // -------------- Create sub-folders for all output ROI-mask and Face-ROI images under the BATCH folder.
        //
        // ROI mask images.
        String frontCheekMaskFolderName = "frontCheekMask";
        String newFrontCheekMaskFolder = null;

        String frontChinMaskFolderName = "frontChinMask";
        String newFrontChinMaskFolder = null;

        String frontForeheadMaskFolderName = "frontForeheadMask";
        String newFrontForeheadMaskFolder = null;

        String frontFullFaceMaskFolderName = "frontFullFaceMask";
        String newFrontFullFaceMaskFolder = null;

        String frontNoseMaskFolderName = "frontNoseMask";
        String newFrontNoseMaskFolder = null;

        String leftCheekMaskFolderName = "leftCheekMask";
        String newLeftCheekMaskFolder = null;

        String leftChinMaskFolderName = "leftChinMask";
        String newLeftChinMaskFolder = null;

        String leftForeheadMaskFolderName = "leftForeheadMask";
        String newLeftForeheadMaskFolder = null;

        String leftFullFaceMaskFolderName = "leftFullFaceMask";
        String newLeftFullFaceMaskFolder = null;

        String leftNoseMaskFolderName = "leftNoseMask";
        String newLeftNoseMaskFolder = null;

        String rightCheekMaskFolderName = "rightCheekMask";
        String newRightCheekMaskFolder = null;

        String rightChinMaskFolderName = "rightChinMask";
        String newRightChinMaskFolder = null;

        String rightForeheadMaskFolderName = "rightForeheadMask";
        String newRightForeheadMaskFolder = null;

        String rightFullFaceMaskFolderName = "rightFullFaceMask";
        String newRightFullFaceMaskFolder = null;

        String rightNoseMaskFolderName = "rightNoseMask";
        String newRightNoseMaskFolder = null;

        try {
            newFrontCheekMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontCheekMaskFolderName);
            newFrontChinMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontChinMaskFolderName);
            newFrontForeheadMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontForeheadMaskFolderName);
            newFrontFullFaceMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontFullFaceMaskFolderName);
            newFrontNoseMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontNoseMaskFolderName);

            newLeftCheekMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, leftCheekMaskFolderName);
            newLeftChinMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, leftChinMaskFolderName);
            newLeftForeheadMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, leftForeheadMaskFolderName);
            newLeftFullFaceMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, leftFullFaceMaskFolderName);
            newLeftNoseMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, leftNoseMaskFolderName);

            newRightCheekMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, rightCheekMaskFolderName);
            newRightChinMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, rightChinMaskFolderName);
            newRightForeheadMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, rightForeheadMaskFolderName);
            newRightFullFaceMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, rightFullFaceMaskFolderName);
            newRightNoseMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, rightNoseMaskFolderName);
            Log.d(TAG, "run: getting folder for 2nd step");
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error creating new front wrinkle mask output folder", e);
        }

        File frontCheekMask = new File(newFrontCheekMaskFolder, "front cheek.jpg");
        File frontChinMask = new File(newFrontChinMaskFolder, "front chin.jpg");
        File frontForeheadMask = new File(newFrontForeheadMaskFolder, "front forehead.jpg");
        File frontFullFace = new File(newFrontFullFaceMaskFolder, "front full face.jpg");
        File frontNose = new File(newFrontNoseMaskFolder, "front nose.jpg");

        String frontCheekMaskOutputPath = frontCheekMask.getAbsolutePath();
        String frontChinMaskOutputPath = frontChinMask.getAbsolutePath();
        String frontForeheadMaskOutputPath = frontForeheadMask.getAbsolutePath();
        String frontFullFaceMaskOutputPath = frontFullFace.getAbsolutePath();
        String frontNoseMaskOutputPath = frontNose.getAbsolutePath();

        File leftCheekMask = new File(newLeftCheekMaskFolder, "left cheek.jpg");
        File leftChinMask = new File(newLeftChinMaskFolder, "left chin.jpg");
        File leftForeheadMask = new File(newLeftForeheadMaskFolder, "left forehead.jpg");
        File leftFullFace = new File(newLeftFullFaceMaskFolder, "left full face.jpg");
        File leftNose = new File(newLeftNoseMaskFolder, "left nose.jpg");

        String leftCheekMaskOutputPath = leftCheekMask.getAbsolutePath();
        String leftChinMaskOutputPath = leftChinMask.getAbsolutePath();
        String leftForeheadMaskOutputPath = leftForeheadMask.getAbsolutePath();
        String leftFullFaceMaskOutputPath = leftFullFace.getAbsolutePath();
        String leftNoseMaskOutputPath = leftNose.getAbsolutePath();

        File rightCheekMask = new File(newRightCheekMaskFolder, "right cheek.jpg");
        File rightChinMask = new File(newRightChinMaskFolder, "right chin.jpg");
        File rightForeheadMask = new File(newRightForeheadMaskFolder, "right forehead.jpg");
        File rightFullFace = new File(newRightFullFaceMaskFolder, "right full face.jpg");
        File rightNose = new File(newRightNoseMaskFolder, "right nose.jpg");

        String rightCheekMaskOutputPath = rightCheekMask.getAbsolutePath();
        String rightChinMaskOutputPath = rightChinMask.getAbsolutePath();
        String rightForeheadMaskOutputPath = rightForeheadMask.getAbsolutePath();
        String rightFullFaceMaskOutputPath = rightFullFace.getAbsolutePath();
        String rightNoseMaskOutputPath = rightNose.getAbsolutePath();

        String frontOilinessROIFolderName = "front oiliness roi";
        String frontOilinessROIFolder = null;

        String frontRadianceROIFolderName = "front radiance roi";
        String frontRadianceROIFolder = null;

        String frontPoresROIFolderName = "front pores roi";
        String frontPoresROIFolder = null;

        try {
            frontOilinessROIFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontOilinessROIFolderName);
            frontRadianceROIFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontRadianceROIFolderName);
            frontPoresROIFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontPoresROIFolderName);
            Log.d(TAG, "run: frontOilinessROIFolder is taken");
        } catch(IOException e){
            Log.e("ImageSegmentation", "Error creating ROI output folders", e);
        }

        File frontOilinessROI = new File(frontOilinessROIFolder, "front oiliness roi.jpg");
        File frontRadianceROI = new File(frontRadianceROIFolder, "front oiliness roi.jpg");
        File frontPoresROI = new File(frontPoresROIFolder, "front pores roi.jpg");

        String frontOilinessROIOutputPath = frontOilinessROI.getAbsolutePath();
        String frontRadianceROIOutputPath = frontRadianceROI.getAbsolutePath();
        String frontPoresROIOutputPath = frontPoresROI.getAbsolutePath();

        // NOTICE!!!
        //
        // OriginalFrontPPLPath -> should be anonymized front PPL Image with only hair and face skin part.
        FFAGetROIs.doAnalysis(MainActivity.this,
                originalFrontPPLPath,
                originalLeftPPLPath,
                originalRightPPLPath,
                roiAIModule,
                frontFullFaceMaskOutputPath,
                leftFullFaceMaskOutputPath,
                rightFullFaceMaskOutputPath,
                frontPoresROIOutputPath,
                frontOilinessROIOutputPath,
                frontRadianceROIOutputPath,
                frontForeheadMaskOutputPath,
                frontNoseMaskOutputPath,
                frontCheekMaskOutputPath,
                frontChinMaskOutputPath,
                leftForeheadMaskOutputPath,
                leftNoseMaskOutputPath,
                leftCheekMaskOutputPath,
                leftChinMaskOutputPath,
                rightForeheadMaskOutputPath,
                rightNoseMaskOutputPath,
                rightCheekMaskOutputPath,
                rightChinMaskOutputPath);

        Log.d(TAG, "run: Time for ROI");
        long roiTime = System.currentTimeMillis() / 1000;
        Log.println(Log.VERBOSE, "Time for ROIs: ", String.valueOf(roiTime - startTime));

        // ------------------------------ (3) Prepare Input Paths for Skin Analyses ------------------------------
        //
        String frontFullFaceMaskInputPath = frontFullFaceMaskOutputPath;
        String leftFullFaceMaskInputPath = leftFullFaceMaskOutputPath;
        String rightFullFaceMaskInputPath = rightFullFaceMaskOutputPath;

        String frontForeheadMaskInputPath = frontForeheadMaskOutputPath;
        String frontNoseMaskInputPath = frontNoseMaskOutputPath;
        String frontCheekMaskInputPath = frontCheekMaskOutputPath;
        String frontChinMaskInputPath = frontChinMaskOutputPath;

        String leftForeheadMaskInputPath = leftForeheadMaskOutputPath;
        String leftNoseMaskInputPath = leftNoseMaskOutputPath;
        String leftCheekMaskInputPath = leftCheekMaskOutputPath;
        String leftChinMaskInputPath = leftChinMaskOutputPath;

        String rightForeheadMaskInputPath = rightForeheadMaskOutputPath;
        String rightNoseMaskInputPath = rightNoseMaskOutputPath;
        String rightCheekMaskInputPath = rightCheekMaskOutputPath;
        String rightChinMaskInputPath = rightChinMaskOutputPath;

        // ------------------------------ (4) FFA Wrinkle & Spots Analysis ------------------------------
        //
        // -------------- Create sub-folders for output mask images and result images for Wrinkles & Spots, under the BATCH folder.
        String frontWrinkleMaskFolderName = "frontWrinkleMask";
        String frontWrinkleResultFolderName = "frontWrinkleResult";
        String newFrontWrinkleMaskFolder = null;
        String newFrontWrinkleResultFolder = null;

        String leftWrinkleMaskFolderName = "leftWrinkleMask";
        String leftWrinkleResultFolderName = "leftWrinkleResult";
        String newLeftWrinkleMaskFolder = null;
        String newLeftWrinkleResultFolder = null;

        String rightWrinkleMaskFolderName = "rightWrinkleMask";
        String rightWrinkleResultFolderName = "rightWrinkleResult";
        String newRightWrinkleMaskFolder = null;
        String newRightWrinkleResultFolder = null;

        String frontSpotsMaskOutputFolderName = "front spots mask";
        String newFrontSpotsMaskOutputFolder = null;

        String leftSpotsMaskOutputFolderName = "left spots mask";
        String newLeftSpotsMaskOutputFolder = null;

        String rightSpotsMaskOutputFolderName = "right spots mask";
        String newRightSpotsMaskOutputFolder = null;

        try {
            newFrontWrinkleMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontWrinkleMaskFolderName);
            newFrontWrinkleResultFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontWrinkleResultFolderName);

            newLeftWrinkleMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, leftWrinkleMaskFolderName);
            newLeftWrinkleResultFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, leftWrinkleResultFolderName);

            newRightWrinkleMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, rightWrinkleMaskFolderName);
            newRightWrinkleResultFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, rightWrinkleResultFolderName);

            newFrontSpotsMaskOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontSpotsMaskOutputFolderName);
            newLeftSpotsMaskOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, leftSpotsMaskOutputFolderName);
            newRightSpotsMaskOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, rightSpotsMaskOutputFolderName);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error creating wrinkle and spots output folders", e);
        }

        File frontWrinkleMask = new File(newFrontWrinkleMaskFolder, "front_wrinkle_mask.jpg");
        File leftWrinkleMask = new File(newLeftWrinkleMaskFolder, "left_wrinkle_mask.jpg");
        File rightWrinkleMask = new File(newRightWrinkleMaskFolder, "right_wrinkle_mask.jpg");

        // --------------- Prepare output paths for Wrinkles & Spots.
        String frontWrinkleMaskOutputPath = frontWrinkleMask.getAbsolutePath();
        String leftWrinkleMaskOutputPath = leftWrinkleMask.getAbsolutePath();
        String rightWrinkleMaskOutputPath = rightWrinkleMask.getAbsolutePath();

        File frontWrinkleResult = new File(newFrontWrinkleResultFolder, "front_wrinkle_result.jpg");
        File leftWrinkleResult = new File(newLeftWrinkleResultFolder, "left_wrinkle_result.jpg");
        File rightWrinkleResult = new File(newRightWrinkleResultFolder, "right_wrinkle_result.jpg");

        String frontWrinkleResultOutputPath = frontWrinkleResult.getAbsolutePath();
        String leftWrinkleResultOutputPath = leftWrinkleResult.getAbsolutePath();
        String rightWrinkleResultOutputPath = rightWrinkleResult.getAbsolutePath();


        File frontSpotsMask = new File(newFrontSpotsMaskOutputFolder, "front spots mask.jpg");
        File leftSpotsMask = new File(newLeftSpotsMaskOutputFolder, "left spots mask.jpg");
        File rightSpotsMask = new File(newRightSpotsMaskOutputFolder, "right spots mask.jpg");

        String frontSpotsMaskOutputPath = frontSpotsMask.getAbsolutePath();
        String leftSpotsMaskOutputPath = leftSpotsMask.getAbsolutePath();
        String rightSpotsMaskOutputPath = rightSpotsMask.getAbsolutePath();

        boolean pigmentationSpotsSideImageEnabled = false;

        FFAWrinklePigmentationSpots100.doAnalysis(MainActivity.this,
                wrinkleSpotsTFModel,
                originalFrontPPLPath,
                originalLeftPPLPath,
                originalRightPPLPath,
                frontFullFaceMaskInputPath,
                leftFullFaceMaskInputPath,
                rightFullFaceMaskInputPath,
                frontWrinkleResultOutputPath,
                leftWrinkleResultOutputPath,
                rightWrinkleResultOutputPath,
                frontWrinkleMaskOutputPath,
                leftWrinkleMaskOutputPath,
                rightWrinkleMaskOutputPath,
                frontSpotsMaskOutputPath,
                leftSpotsMaskOutputPath,
                rightSpotsMaskOutputPath,
                pigmentationSpotsSideImageEnabled);

        long wrinkleSpotsTime = System.currentTimeMillis() / 1000;
        Log.println(Log.VERBOSE, "Time for wrinkles & spots: ", String.valueOf(wrinkleSpotsTime - startTime));

        // ------------------------------ (5) Dark Circle Analysis ------------------------------
        //
        String darkCircleMaskFolderName = "darkCircleMask";
        String darkCircleResultFolderName = "darkCircleResult";
        String newDarkCircleMaskFolder = null;
        String newDarkCircleResultFolder = null;

        try {
            newDarkCircleMaskFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, darkCircleMaskFolderName);
            newDarkCircleResultFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, darkCircleResultFolderName);
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error creating dark circle output folders", e);
        }

        File darkCircleMask = new File(newDarkCircleMaskFolder, "dark_circle_mask.jpg");
        String darkCircleMaskOutputPath = darkCircleMask.getAbsolutePath();

        File darkCircleResult = new File(newDarkCircleResultFolder, "dark_circle_result.jpg");
        String darkCircleResultOutputPath = darkCircleResult.getAbsolutePath();

        FFADarkCircle100.doAnalysis(MainActivity.this,
                darkCircleTFModel,
                originalFrontPPLPath,
                frontFullFaceMaskInputPath,
                darkCircleResultOutputPath,
                darkCircleMaskOutputPath);

        long darkCircleTime = System.currentTimeMillis() / 1000;
        Log.println(Log.VERBOSE, "Time for dark circle: ", String.valueOf(darkCircleTime - startTime));

        // ------------------------------ (6) Test Image Processing Algorithms ------------------------------
        //
        JNIFFAImageProCW myFFAImgProc = new JNIFFAImageProCW();
        // ----- Pores algorithm -----
        //
        String poresMaskOutputFolderName = "pores mask";
        String newPoresMaskOutputFolder = null;

        String poresResultOutputFolderName = "pores result";
        String newPoresResultOutputFolder = null;

        try {
            newPoresMaskOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, poresMaskOutputFolderName);
            newPoresResultOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, poresResultOutputFolderName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating Pores output folders", e);
        }

        File poresMask = new File(newPoresMaskOutputFolder, "pores mask.jpg");
        File poresResult = new File(newPoresResultOutputFolder, "pores result.jpg");

        String poresMaskOutputPath = poresMask.getAbsolutePath();
        String poresResultOutputPath = poresResult.getAbsolutePath();
        String poresROIInputPath = frontPoresROIOutputPath;

        String poresResStr = myFFAImgProc.FFALocalPores100Jni(
                poresROIInputPath,
                originalFrontPPLPath,
                poresResultOutputPath,
                poresMaskOutputPath);

        System.out.println("Returned Pores Result is: " + poresResStr);

        // ----- Oiliness algorithm -----
        //
        String oilinessRoiImgPath = frontOilinessROIOutputPath;

        String oilinessResultOutputFolderName = "oiliness result";
        String newOilinessResultOutputFolder = null;

        String oilinessGreenMaskOutputFolderName = "oiliness green mask";
        String newOilinessGreenMaskOutputFolder = null;

        String oilinessWhiteMaskOutputFolderName = "oiliness white mask";
        String newOilinessWhiteMaskOutputFolder = null;

        try {
            newOilinessResultOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, oilinessResultOutputFolderName);
            newOilinessGreenMaskOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, oilinessGreenMaskOutputFolderName);
            newOilinessWhiteMaskOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, oilinessWhiteMaskOutputFolderName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating Oiliness output folders", e);
        }

        File oilinessResult = new File(newOilinessResultOutputFolder, "oiliness result.jpg");
        File oilinessGreenMask = new File(newOilinessGreenMaskOutputFolder, "oiliness green mask.jpg");
        File oilinessWhiteMask = new File(newOilinessWhiteMaskOutputFolder, "oiliness white mask.jpg");

        String oilinessResultOutputPath = oilinessResult.getAbsolutePath();
        String oilinessGreenMaskOutputPath = oilinessGreenMask.getAbsolutePath();
        String oilinessWhiteMaskOutputPath = oilinessWhiteMask.getAbsolutePath();

        String oilinessResStr = myFFAImgProc.FFALocalOiliness100Jni(
                originalFrontPPLPath,
                oilinessRoiImgPath,
                oilinessResultOutputPath,
                oilinessGreenMaskOutputPath,
                oilinessWhiteMaskOutputPath);

        System.out.println("Returned oiliness string is: " + oilinessResStr);

        // ----- Radiance & Dullness algorithm -----
        //
        String radianceRoiImgPath = frontRadianceROIOutputPath;

        String radianceResultOutputFolderName = "radiance result";
        String newRadianceResultOutputFolder = null;

        String radianceGrayMaskOutputFolderName = "radiance gray mask";
        String newRadianceGrayMaskOutputFolder = null;

        String radianceWhiteMaskOutputFolderName = "radiance white mask";
        String newRadianceWhiteMaskOutputFolder = null;

        try {
            newRadianceResultOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, radianceResultOutputFolderName);
            newRadianceGrayMaskOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, radianceGrayMaskOutputFolderName);
            newRadianceWhiteMaskOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, radianceWhiteMaskOutputFolderName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating Radiance output folders", e);
        }

        File radianceResult = new File(newRadianceResultOutputFolder, "radiance result.jpg");
        File radianceGrayMask = new File(newRadianceGrayMaskOutputFolder, "radiance gray mask.jpg");
        File radianceWhiteMask = new File(newRadianceWhiteMaskOutputFolder, "radiance white mask.jpg");

        String radianceResultOutputPath = radianceResult.getAbsolutePath();
        String radianceGrayMaskOutputPath = radianceGrayMask.getAbsolutePath();
        String radianceWhiteMaskOutputPath = radianceWhiteMask.getAbsolutePath();

        String radianceDullnessResStr = myFFAImgProc.FFALocalRadianceDullness100Jni(
                originalFrontPPLPath,
                radianceRoiImgPath,
                radianceResultOutputPath,
                radianceGrayMaskOutputPath,
                radianceWhiteMaskOutputPath);

        System.out.println("Returned Radiance & Dullness String is: " + radianceDullnessResStr);

        // ----- Get spots indexing and results. -----
        String frontSpotsResultOutputFolderName = "front spots result";
        String newFrontSpotsResultOutputFolder = null;

        String leftSpotsResultOutputFolderName = "left spots result";
        String newLeftSpotsResultOutputFolder = null;

        String rightSpotsResultOutputFolderName = "right spots result";
        String newRightSpotsResultOutputFolder = null;

        try {
            newFrontSpotsResultOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, frontSpotsResultOutputFolderName);
            newLeftSpotsResultOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, leftSpotsResultOutputFolderName);
            newRightSpotsResultOutputFolder = MyUtil.newFolderPath(getApplicationContext(), parentFolder, rightSpotsResultOutputFolderName);
        } catch(IOException e) {
            Log.e("ImageSegmentation", "Error creating Pigmentation/Spots output folders", e);
        }

        File frontSpotsResult = new File(newFrontSpotsResultOutputFolder, "front spots result.jpg");
        File leftSpotsResult = new File(newLeftSpotsResultOutputFolder, "left spots result.jpg");
        File rightSpotsResult = new File(newRightSpotsResultOutputFolder, "right spots result.jpg");

        String frontSpotsResultOutputPath = frontSpotsResult.getAbsolutePath();
        String leftSpotsResultOutputPath = leftSpotsResult.getAbsolutePath();
        String rightSpotsResultOutputPath = rightSpotsResult.getAbsolutePath();

        String frontFullSpotsMaskInputPath = frontSpotsMaskOutputPath;
        String leftFullSpotsMaskInputPath = leftSpotsMaskOutputPath;
        String rightFullSpotsMaskInputPath = rightSpotsMaskOutputPath;

        String spotsRetStr = myFFAImgProc.FFALocalPigmentationSpots100Jni(
                originalFrontPPLPath,
                originalLeftPPLPath,
                originalRightPPLPath,
                frontSpotsResultOutputPath,
                leftSpotsResultOutputPath,
                rightSpotsResultOutputPath,
                frontForeheadMaskInputPath,
                frontNoseMaskInputPath,
                frontCheekMaskInputPath,
                frontChinMaskInputPath,
                frontFullFaceMaskInputPath,
                leftForeheadMaskInputPath,
                leftNoseMaskInputPath,
                leftCheekMaskInputPath,
                leftChinMaskInputPath,
                leftFullFaceMaskInputPath,
                rightForeheadMaskInputPath,
                rightNoseMaskInputPath,
                rightCheekMaskInputPath,
                rightChinMaskInputPath,
                rightFullFaceMaskInputPath,
                frontFullSpotsMaskInputPath,
                leftFullSpotsMaskInputPath,
                rightFullSpotsMaskInputPath,
                pigmentationSpotsSideImageEnabled);

        System.out.println("Returned Spots String is: " + spotsRetStr);

        long IPtime = System.currentTimeMillis() / 1000;
        Log.println(Log.VERBOSE, "Time for all image processing algorithms: ", String.valueOf(IPtime - startTime));

        // ---- Test elasticity -----
        //


        // ------------------------------ Finishing ------------------------------
        //
        // Processing results, and local files etc.

        long endTime = System.currentTimeMillis() / 1000;
        Log.println(Log.VERBOSE, "Time for FFA overall analysis: ", String.valueOf(endTime - startTime));

        runOnUiThread(new Runnable(){
            @Override
            public void run() {

            }
        });
    }
}