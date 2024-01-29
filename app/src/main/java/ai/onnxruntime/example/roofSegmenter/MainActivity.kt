package ai.onnxruntime.example.superresolution

import ai.onnxruntime.*
import ai.onnxruntime.example.roofSegmenter.RoofSegmenter
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.drawable.toBitmap
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.util.*


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var inputImage: ImageButton? = null
    private var outputImage: ImageView? = null
    private var segmentRoofButton: Button? = null

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImage = findViewById(R.id.inputImage)
        outputImage = findViewById(R.id.imageView)
        segmentRoofButton = findViewById(R.id.segment_Roof_Button)

        // Initialize Ort Session and register the onnxruntime extensions package.
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        segmentRoofButton?.setOnClickListener {
            try {
                performRoofSegmentation(ortSession)
                Toast.makeText(baseContext, "Segmented roof successfully!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e("Exception caught while segmenting roof", e.toString())
                Toast.makeText(baseContext, "Failed to segment roof", Toast.LENGTH_SHORT)
                    .show()
            }
        }

        inputImage?.setOnClickListener {
            pickImageFromGallery()
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

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    // Read the nanoroof.onnx model from the raw resources folder.
    private fun readModel(): ByteArray {
        val modelID = R.raw.nanoroof
        return resources.openRawResource(modelID).readBytes()
    }

    private fun performRoofSegmentation(ortSession: OrtSession) {
        var roofSegmenter = RoofSegmenter()
        var result = inputImage?.drawable?.toBitmap()?.let { roofSegmenter.segmentRoof(this, it, ortEnv, ortSession) }
        if (result != null) {
            outputImage?.setImageBitmap(result.outputBitmap)
        };
    }
}
