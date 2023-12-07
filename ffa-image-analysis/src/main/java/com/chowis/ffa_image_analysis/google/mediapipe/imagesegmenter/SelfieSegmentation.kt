package com.google.mediapipe.imagesegmenter

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.util.Log
import android.widget.Toast
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Core.bitwise_and
import org.opencv.core.CvType.CV_8UC1
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgcodecs.Imgcodecs.imwrite
import org.opencv.imgproc.Imgproc.COLOR_BGR2RGB
import org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR
import org.opencv.imgproc.Imgproc.COLOR_RGB2BGR
import org.opencv.imgproc.Imgproc.INTER_AREA
import org.opencv.imgproc.Imgproc.INTER_LINEAR
import org.opencv.imgproc.Imgproc.cvtColor
import org.opencv.imgproc.Imgproc.resize
import java.io.File
import java.nio.ByteBuffer
import java.util.*

class SelfieSegmentation(
    private val context: Context
) : ImageSegmenterHelper.SegmenterListener {

    private lateinit var imageSegmenterHelper: ImageSegmenterHelper
    private var backgroundScope: CoroutineScope? = null
    private var fixedRateTimer: Timer? = null

    // Load and segment the image and get anonymized hair + face mask.
    fun runSegmentationOnImage(frontPPL : File,
                               leftPPL : File,
                               rightPPL : File,
                               inputFrontPPLPath: String, outputFrontPPLPath: String,
                               inputLeftPPLPath: String, outputLeftPPLPath: String,
                               inputRightPPLPath: String, outputRightPPLPath: String) {

        // Configure coroutine and AI model.
        backgroundScope = CoroutineScope(Dispatchers.IO)

        imageSegmenterHelper = ImageSegmenterHelper(
            //context = requireContext(),
            context = context,
            runningMode = RunningMode.IMAGE,
            currentModel = ImageSegmenterHelper.MODEL_SELFIE_MULTICLASS,
            currentDelegate = ImageSegmenterHelper.DELEGATE_CPU,
            imageSegmenterListener = this
        )

        var inputImageFrontPPL = toBitmap(frontPPL)
        inputImageFrontPPL = inputImageFrontPPL.scaleDown(INPUT_IMAGE_MAX_WIDTH)

        var inputImageLeftPPL = toBitmap(leftPPL)
        inputImageLeftPPL = inputImageLeftPPL.scaleDown(INPUT_IMAGE_MAX_WIDTH)

        var inputImageRightPPL = toBitmap(rightPPL)
        inputImageRightPPL = inputImageRightPPL.scaleDown(INPUT_IMAGE_MAX_WIDTH)

        // Run image segmentation on the input image
        backgroundScope?.launch {
            val mpImageFrontPPL = BitmapImageBuilder(inputImageFrontPPL).build()
            val resultFrontPPL = imageSegmenterHelper?.segmentImageFile(mpImageFrontPPL)

            val mpImageLeftPPL = BitmapImageBuilder(inputImageLeftPPL).build()
            val resultLeftPPL = imageSegmenterHelper?.segmentImageFile(mpImageLeftPPL)

            val mpImageRightPPL = BitmapImageBuilder(inputImageRightPPL).build()
            val resultRightPPL = imageSegmenterHelper?.segmentImageFile(mpImageRightPPL)

            Log.println(Log.VERBOSE, "selfie multi class", "---------- segmented ----------")

            // process AI output results.
            getAnonymizedMask(resultFrontPPL!!, inputFrontPPLPath, outputFrontPPLPath)

            getAnonymizedMask(resultLeftPPL!!, inputLeftPPLPath, outputLeftPPLPath)

            getAnonymizedMask(resultRightPPL!!, inputRightPPLPath, outputRightPPLPath)
        }
    }

    // convert Uri/assets to bitmap image.
    private fun toBitmap(file : File): Bitmap {
        val source = ImageDecoder.createSource(file)
        return (ImageDecoder.decodeBitmap(source)).copy(Bitmap.Config.ARGB_8888, true)
    }

    /**
     * Scales down the given bitmap to the specified target width while maintaining aspect ratio.
     * If the original image is already smaller than the target width, the original image is returned.
     */
    private fun Bitmap.scaleDown(targetWidth: Float): Bitmap {
        // if this image smaller than widthSize, return original image
        Log.println(Log.VERBOSE, "original image width:", width.toString())

        if (targetWidth >= width) return this
        val scaleFactor = targetWidth / width
        return Bitmap.createScaledBitmap(
            this,
            (width * scaleFactor).toInt(),
            (height * scaleFactor).toInt(),
            false
        )
    }

    private fun getAnonymizedMask(result: ImageSegmenterResult, inputPath : String, outputPath: String) {
        val newImage = result.categoryMask().get()

        val scaledWidth = newImage.width
        val scaledHeight = newImage.height
        val byteBuffer : ByteBuffer = ByteBufferExtractor.extract(newImage)

        val originalImg : Mat = imread(inputPath)
        val width = originalImg.width()
        val height = originalImg.height()

        // Create the mask for hair (category 1) and face skin (category 3).
        var mpMask : Mat = Mat.zeros(scaledHeight, scaledWidth, CV_8UC1)

        for (i in 0 until scaledHeight) {
            for (j in 0 until scaledWidth) {
                // Using unsigned int here because selfie segmentation returns 0 or 255U (-1 signed)
                // with 0 being the found person, 255U for no label.
                // Deeplab uses 0 for background and other labels are 1-19,
                // so only providing 20 colors from ImageSegmenterHelper -> labelColors
                val category = (byteBuffer.get(i * scaledWidth + j).toUInt() % 20U).toInt()

                if (category == 1 || category == 3) {
                    mpMask.put(i, j, 255.toDouble())
                } else {
                    mpMask.put(i, j, 0.toDouble())
                }
            }
        }

        // resize mask image to original size.
        if (width * height > scaledWidth * scaledHeight) {
            resize(mpMask, mpMask, Size(width.toDouble(), height.toDouble()), INTER_LINEAR.toDouble())
        }
        if (width * height < scaledWidth * scaledHeight) {
            resize(mpMask, mpMask, Size(width.toDouble(), height.toDouble()), INTER_AREA.toDouble())
        }

        var anonymizedImg = Mat()
        cvtColor(mpMask, mpMask, COLOR_GRAY2BGR)
        bitwise_and(mpMask, originalImg, anonymizedImg)

        // save image.
        //MyUtil.saveMatToGallery(context, "anonymized image", "anonymized front ppl image", anonymizedImg)
        imwrite(outputPath, anonymizedImg)
    }

    private fun stopAllTasks() {
        // cancel all jobs
        fixedRateTimer?.cancel()
        fixedRateTimer = null
        backgroundScope?.cancel()
        backgroundScope = null

        // clear Image Segmenter
        imageSegmenterHelper?.clearListener()
        imageSegmenterHelper?.clearImageSegmenter()
        //imageSegmenterHelper = null
    }

    private fun segmentationError() {
        stopAllTasks()
    }

    override fun onError(error: String, errorCode: Int) {
        backgroundScope?.launch {
            withContext(Dispatchers.Main) {
                segmentationError()
                Toast.makeText(context, error, Toast.LENGTH_SHORT)
                    .show()
                /*if (errorCode == ImageSegmenterHelper.GPU_ERROR) {
                    fragmentGalleryBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                        ImageSegmenterHelper.DELEGATE_CPU, false
                    )
                }*/
            }
        }
    }

    override fun onResults(resultBundle: ImageSegmenterHelper.ResultBundle) {
        TODO("Not yet implemented")
    }

    companion object {
        private const val INPUT_IMAGE_MAX_WIDTH = 512F
    }
}