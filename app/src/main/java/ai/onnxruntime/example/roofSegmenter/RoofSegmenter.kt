package ai.onnxruntime.example.roofSegmenter

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.exp
import kotlin.math.roundToInt

internal data class Result(
    var outputBitmap: Bitmap? = null,
    var onnxTime: Float? = null,
    var fullTime: Float? = null,
)

internal class RoofSegmenter(
) {
    //Creates a FloatBuffer that meets the requirements of the model input tensor and populates it with the pixel values of the input image
    //The model input tensor is of shape (1, 3, 640, 640) and is of type float32 (CV_32F) with values between 0 and 1
    //1 is the batch size, 3 is the number of channels (RGB), 640 is the width and height of the input image
    private fun floatBufferFromMat(mat: Mat): FloatBuffer {
        val floatArray = FloatArray(mat.width()*mat.height()*mat.channels())
        mat.convertTo(mat, CvType.CV_32F)

        for (channel in 0 until 3) {
            for (row in 0 until 640) {
                for (col in 0 until 640) {
                    val value = mat.get(row, col)[channel].toFloat()
                    val index = channel * 409600 + row * 640 + col
                    floatArray[index] = value/255 //Normalize the pixel values to be between 0 and 1
                }
            }
        }
        return FloatBuffer.wrap(floatArray)
    }

    // Transposes a 2D array of floats
    private fun transpose2DArray(inputArray: Array<Array<Float>>): Array<Array<Float>> {
        val numRows = inputArray.size
        val numCols = inputArray[0].size

        val transposedArray = Array(numCols) { Array(numRows) { 0f } }

        for (i in 0 until numRows) {
            for (j in 0 until numCols) {
                transposedArray[j][i] = inputArray[i][j]
            }
        }
        return transposedArray
    }

    // Calculates the intersection of two bounding boxes
    private fun intersection(box1: FloatArray, box2: FloatArray): Float {
        val (box1x1, box1y1, box1x2, box1y2) = box1.take(4)
        val (box2x1, box2y1, box2x2, box2y2) = box2.take(4)
        val x1 = maxOf(box1x1, box2x1)
        val y1 = maxOf(box1y1, box2y1)
        val x2 = minOf(box1x2, box2x2)
        val y2 = minOf(box1y2, box2y2)
        return if (x1 < x2 && y1 < y2) (x2 - x1) * (y2 - y1) else 0f
    }

    // Calculates the union of two bounding boxes
    private fun union(box1: FloatArray, box2: FloatArray): Float {
        val (box1x1, box1y1, box1x2, box1y2) = box1.take(4)
        val (box2x1, box2y1, box2x2, box2y2) = box2.take(4)
        val box1area = (box1x2 - box1x1) * (box1y2 - box1y1)
        val box2area = (box2x2 - box2x1) * (box2y2 - box2y1)
        return box1area + box2area - intersection(box1, box2)
    }

    // Calculates the intersection over union of two bounding boxes
    private fun iou(box1: FloatArray, box2: FloatArray): Float {
        val intersectionArea = intersection(box1, box2)
        val unionArea = union(box1, box2)
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    // Applies a sigmoid function to the mask pixel to keep values between 0 and 1 and represent its probability
    // Then checks if it is above the maskThreshold
    private fun sigThreshold(x: Float, maskThreshold: Float): Boolean {
        val sig = 1 / (1 + exp(-x).toFloat())
        return sig > maskThreshold
    }

    // Creates a mask of booleans of the selected bounding box by creating a mask cutout and applying the sigThreshold to each pixel
    private fun getMask(flatMask: FloatArray, box: FloatArray, maskThreshold: Float): Array<BooleanArray> {
        val (x1, y1, x2, y2) = box.map { it.toInt() }
        val mask = Array(y2 - y1) { i ->
            BooleanArray(x2 - x1) { j ->
                sigThreshold(flatMask[(i + y1) * 160 + (j + x1)], maskThreshold)
            }
        }
        return mask
    }

    // Creates an overlay bitmap from the mask with specified color and alpha to lay over the original image
    private fun createMaskOverlayBitmap(mask: Array<BooleanArray>, rgba: IntArray): Bitmap {
        val width = mask[0].size
        val height = mask.size

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (i in 0 until height) {
            for (j in 0 until width) {
                if(mask[i][j]) {
                    bitmap.setPixel(j, i, Color.argb(rgba[3], rgba[0], rgba[1], rgba[2]))
                } else {
                    bitmap.setPixel(j, i, Color.argb(0, 0, 0, 0))
                }
            }
        }
        return bitmap
    }

    // Overlays the mask bitmap on top of the original image
    private fun overlayBitmaps(baseBitmap: Bitmap, overlayBitmap: Bitmap): Bitmap {
        val resultBitmap = Bitmap.createBitmap(baseBitmap.width, baseBitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultBitmap)

        val paint = Paint()
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)

        // Draw the baseBitmap
        canvas.drawBitmap(baseBitmap, 0f, 0f, null)

        // Draw the overlayBitmap on top of baseBitmap
        canvas.drawBitmap(overlayBitmap, 0f, 0f, paint)

        return resultBitmap
    }

    // Main function that performs roof segmentation
    fun segmentRoof(context: Context, bitmap: Bitmap, ortEnv: OrtEnvironment, ortSession: OrtSession): Result? {
        val fullStartTime = System.currentTimeMillis() // Start measuring the runtime of the whole function
        var result: Result? = Result() //Initialize instance of Result class

        val options = Options(context)

        OpenCVLoader.initDebug() // Initialize OpenCV. This is required for the bitmap to mat conversion!

        //Store original size to scale the output bitmap back to the original size
        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        // The inpuut of the model ist fixed to 640x640
        val b = Bitmap.createScaledBitmap(bitmap, 640, 640, false)

        val mat = Mat()
        Utils.bitmapToMat(b, mat)

        b.recycle() // Recycle the scaled bitmap to free up memory

        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB) // Convert the bitmap to RGB, as the model expects RGB input

        val floatBuffer = floatBufferFromMat(mat)

        //This is the format the model expects the input tensor to be in
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            floatBuffer,
            longArrayOf(1, 3, 640, 640)
        )

        //Classes: roof, solar-panel, window
        val classAmount = 3

        inputTensor.use {
            val startOnnxTime = System.currentTimeMillis() // Start measuring the runtime of the model
            val output = ortSession.run(Collections.singletonMap("images", inputTensor)) // Run the model!
            val endOnnxTime = System.currentTimeMillis() // Stop measuring the runtime of the model

            var maskLayers = Array(32) { FloatArray(25600) { 0f } }
            var boxes = Array(8400) { FloatArray(classAmount + 4) { 0f } }
            var maskWeights = Array(8400) { FloatArray(32) { 0f } }

            output.use {
                //Get outputs with shape output0: (36 + classAmount, 8400) and output1: (32, 160, 160)
                val output0 = (output?.get(0)?.value as? Array<*>)?.get(0) as Array<FloatArray>
                val output1 = (output?.get(1)?.value as? Array<*>)?.get(0) as Array<Array<FloatArray>>

                //Transpose output0 to shape (8400, 36 + classAmount)hat the 8400 detections are in the first dimension
                var output0Transposed = transpose2DArray(output0.map { it.toTypedArray() }.toTypedArray())

                //Reshape output1 to shape (32, 25600) so that each layer is one-dimensional
                maskLayers = output1.map { plane ->
                    plane.flatMap { row -> row.toList() }.toFloatArray()
                }.toTypedArray()

                // Split output0 into boxes and masks:
                // boxes: (8400, 4 + classAmount) with each entry being xCenter, yCenter, width, height, confidencesOfDetections
                // masks: (8400, 32) with each entry being the factor to multiply the corresponding mask layer to get the correct mask values in the bounding box
                boxes = output0Transposed.map { it.copyOfRange(0, 4 + classAmount).toFloatArray() }.toTypedArray()
                maskWeights = output0Transposed.map { it.copyOfRange(4 + classAmount, 36 + classAmount).toFloatArray() }.toTypedArray()
            }
            output.close() // Close the output to free up memory

            val detections = ArrayList<FloatArray>() // Stores the detections with confidence > box_threshold and their corresponding mask index in croppedMasks
            val croppedMasks = ArrayList<Array<BooleanArray>>() // Stores the cropped masks
            for (i in boxes.indices) {
                val box = boxes[i]
                if(box[4] > options.box_threshold) { // Only consider detections with confidence in class roof > options.box_threshold
                    // converting bounding box from xc, yx, width, height to x1, y1, x2, y2 and scaling from 640x640 to 160x160
                    val x1 = ((box[0]-box[2]/2)/640*160).roundToInt().toFloat()
                    val x2 = ((box[0]+box[2]/2)/640*160).roundToInt().toFloat()
                    val y1 = ((box[1]-box[3]/2)/640*160).roundToInt().toFloat()
                    val y2 = ((box[1]+box[3]/2)/640*160).roundToInt().toFloat()
                    val maskShape = FloatArray(25600)

                    // Multiply the mask layers with the corresponding mask weight and add them up to get the mask
                    for (layerIndex in 0 until 32) {
                        for (pixelIndex in 0 until 25600) {
                            maskShape[pixelIndex] += maskLayers[layerIndex][pixelIndex] * maskWeights[i][layerIndex]
                        }
                    }
                    //Storing detection as: x1, y1, x2, y2, confidence, mask index (current size of croppedMasks)
                    detections.add(floatArrayOf(x1, y1, x2, y2, box[4], croppedMasks.size.toFloat()))
                    //Storing the cropped mask
                    croppedMasks.add(getMask(maskShape, floatArrayOf(x1, y1, x2, y2), options.mask_threshold))
                }
            }
            // If nothing was detected, return an empty result
            if(detections.isEmpty()) {
                return null
            }
            // Sort the detections by confidence in roof class
            detections.sortByDescending { it[4] }

            // Create a full mask from the cropped masks
            val fullMask = Array(160) { _ ->
                BooleanArray(160) { _ ->
                    false
                }
            }
            //Now all the masks are getting combined to one mask for the image
            while (detections.isNotEmpty()) {
                val currentDet = detections[0].copyOf() // Get the detection with the highest confidence
                val mask = croppedMasks[currentDet[5].toInt()] // Get the corresponding mask

                // Add the mask to the full mask at the correct position.
                for (row in mask.indices) {
                    for (column in 0 until mask[0].size) {
                        if(options.merge_masks) {
                            // Using or operator to combine overlapping masks
                            val currentState = fullMask[row+currentDet[1].toInt()][column+currentDet[0].toInt()]
                            fullMask[row+currentDet[1].toInt()][column+currentDet[0].toInt()] = currentState || mask[row][column]
                        } else {
                            // Overwriting overlapping masks
                            fullMask[row+currentDet[1].toInt()][column+currentDet[0].toInt()] = mask[row][column]
                        }
                    }
                }

                // Remove all detections that have an IoU > 0.7 with the current detection to eliminate overlapping detections. This includes the current detection itself.
                detections.retainAll { iou(it, currentDet) < 0.7 }
            }
            val fullEndTime = System.currentTimeMillis() // Stop measuring the runtime of the whole function
            //Add the full mask to the result by scaling the mask back to the original size and overlaying it on top of the original image
            result?.outputBitmap = overlayBitmaps(bitmap,
                Bitmap.createScaledBitmap(
                    createMaskOverlayBitmap(fullMask, intArrayOf(
                        Color.red(options.color?: 0),
                        Color.green(options.color?: 0),
                        Color.blue(options.color?: 0),
                        Color.alpha(options.color?: 0)
                    )
                ), originalWidth, originalHeight, false)
            )
            result?.onnxTime = (endOnnxTime - startOnnxTime) / 1000f // Calculate the runtime of the model
            result?.fullTime = (fullEndTime - fullStartTime) / 1000f // Calculate the runtime of the whole function
        }
        return result
    }
}