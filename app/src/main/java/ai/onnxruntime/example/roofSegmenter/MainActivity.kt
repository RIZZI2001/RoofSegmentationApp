package ai.onnxruntime.example.roofSegmenter

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.KeyEvent
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDialog
import androidx.appcompat.widget.SwitchCompat
import androidx.core.graphics.drawable.toBitmap
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.util.*


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var inputImage: ImageButton? = null
    private var outputImage: ImageView? = null
    private var optionsButton: Button? = null
    private var segmentRoofButton: Button? = null
    private var exportButton: Button? = null
    private lateinit var options: Options
    private var optionsPanel: OptionsPanel? = null

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        options = Options(this)
        inputImage = findViewById(R.id.inputImage)
        outputImage = findViewById(R.id.imageView)
        optionsButton = findViewById(R.id.options_Button)
        segmentRoofButton = findViewById(R.id.segment_Roof_Button)
        exportButton = findViewById(R.id.export_Button)

        // Initialize Ort Session and register the onnxruntime extensions package.
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        optionsButton?.setOnClickListener {
            optionsPanel = OptionsPanel(this, options)
            optionsPanel?.show()
        }
        segmentRoofButton?.setOnClickListener {
            if(inputImage?.drawable == null) {
                Toast.makeText(baseContext, "Please select an image first", Toast.LENGTH_SHORT)
                    .show()
            } else {
                try {
                    performRoofSegmentation(ortSession)
                } catch (e: Exception) {
                    Log.e("Exception caught while segmenting roof", e.toString())
                    Toast.makeText(baseContext, "Failed to segment roof", Toast.LENGTH_SHORT)
                        .show()
                }
            }
        }
        inputImage?.setImageDrawable(getDrawable(R.drawable.default_image))
        inputImage?.setOnClickListener {
            pickImageFromGallery()
        }
        exportButton?.setOnClickListener {
            if(outputImage?.drawable == null) {
                Toast.makeText(baseContext, "Please segment a roof first", Toast.LENGTH_SHORT)
                    .show()
            } else {
                exportImageToGallery()
            }
        }
    }

    // Pick image from gallery
    private fun pickImageFromGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI)
        loadImg.launch(intent)
    }

    private val loadImg =
        registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) {
            if (it.resultCode == Activity.RESULT_OK) {
                inputImage?.setImageURI(it.data?.data)
            }
        }

    private fun exportImageToGallery() {
        val bitmap = outputImage?.drawable?.toBitmap()
        MediaStore.Images.Media.insertImage(
            contentResolver,
            bitmap,
            UUID.randomUUID().toString() + ".png",
            "Roof Segmentation"
        )
        Toast.makeText(baseContext, "Image saved to gallery", Toast.LENGTH_SHORT)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    // Read the nanoroof.onnx model from the raw resources folder.
    private fun readModel(): ByteArray {
        val modelID = R.raw.best
        return resources.openRawResource(modelID).readBytes()
    }

    private fun performRoofSegmentation(ortSession: OrtSession) {
        var roofSegmenter = RoofSegmenter()
        var result = inputImage?.drawable?.toBitmap()?.let { roofSegmenter.segmentRoof(this, it, ortEnv, ortSession) }
        if (result != null) {
            outputImage?.setImageBitmap(result.outputBitmap)
            Toast.makeText(baseContext, "Success! Time: ${result?.fullTime} (${result?.onnxTime})", Toast.LENGTH_SHORT)
                .show()
        } else {
            Toast.makeText(baseContext, "No roofs were found.", Toast.LENGTH_SHORT)
                .show()
            outputImage?.setImageBitmap(null)
        }
    }
}
