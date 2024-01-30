package ai.onnxruntime.example.roofSegmenter

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.view.KeyEvent
import android.view.View
import android.view.WindowManager
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatDialog
import androidx.appcompat.widget.SwitchCompat

class OptionsPanel(
    context: Context,
    private val options: Options,
) : Dialog(context) {
    private lateinit var switchMergeMasks: SwitchCompat
    private lateinit var seekBarRed: SeekBar
    private lateinit var seekBarGreen: SeekBar
    private lateinit var seekBarBlue: SeekBar
    private lateinit var seekBarAlpha: SeekBar
    private lateinit var colorPreview: View
    private lateinit var boxThresholdText: TextView
    private lateinit var boxThresholdBar: SeekBar
    private lateinit var maskThresholdText: TextView
    private lateinit var maskThresholdBar: SeekBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        val dialog = AppCompatDialog(context)
        dialog.setContentView(R.layout.options_menu)
        dialog.setCancelable(true)
        dialog.window?.attributes?.width = WindowManager.LayoutParams.MATCH_PARENT

        switchMergeMasks = dialog.findViewById(R.id.merge_masks_switch)!!

        switchMergeMasks?.isChecked = options.merge_masks
        switchMergeMasks?.setOnCheckedChangeListener { _, isChecked ->
            options.merge_masks = isChecked
        }

        seekBarRed = dialog.findViewById(R.id.seekBarRed)!!
        seekBarGreen = dialog.findViewById(R.id.seekBarGreen)!!
        seekBarBlue = dialog.findViewById(R.id.seekBarBlue)!!
        seekBarAlpha = dialog.findViewById(R.id.seekBarAlpha)!!
        colorPreview = dialog.findViewById(R.id.colorPreview)!!

        setupColorPicker(options.color)

        boxThresholdText = dialog.findViewById(R.id.BoxThresholdText)!!
        boxThresholdBar = dialog.findViewById(R.id.seekBarBoxThreshold)!!
        maskThresholdText = dialog.findViewById(R.id.MaskThresholdText)!!
        maskThresholdBar = dialog.findViewById(R.id.seekBarMaskThreshold)!!

        setupThresholds(options.box_threshold, options.mask_threshold)

        // Handle back button press to dismiss the dialog
        dialog.setOnKeyListener { _, keyCode, _ ->
            if (keyCode == KeyEvent.KEYCODE_BACK) {
                dialog.dismiss()
                true
            } else {
                false
            }
        }
        dialog.show()
    }

    private fun setupColorPicker(color: Int?) {
        seekBarRed.progress = Color.red(color?: 0)
        seekBarGreen.progress = Color.green(color?: 0)
        seekBarBlue.progress = Color.blue(color?: 0)
        seekBarAlpha.progress = Color.alpha(color?: 0)

        updateColorPreview()
        setupSeekBarListener(seekBarRed, "color")
        setupSeekBarListener(seekBarGreen, "color")
        setupSeekBarListener(seekBarBlue, "color")
        setupSeekBarListener(seekBarAlpha, "color")
    }

    private fun setupThresholds(box: Float?, mask: Float?) {
        val boxInt = box?.times(100)?.toInt()
        val maskInt = mask?.times(100)?.toInt()

        boxThresholdBar.progress = boxInt?: 0
        maskThresholdBar.progress = maskInt?: 0

        updateBoxThreshold()
        updateMaskThreshold()
        setupSeekBarListener(boxThresholdBar, "box")
        setupSeekBarListener(maskThresholdBar, "mask")
    }

    private fun updateColorPreview() {
        val color = Color.argb(
            seekBarAlpha.progress,
            seekBarRed.progress,
            seekBarGreen.progress,
            seekBarBlue.progress
        )
        options.color = color   // Save the color to the options
        colorPreview.setBackgroundColor(color)
    }

    private fun updateBoxThreshold() {
        val box = boxThresholdBar.progress.toFloat().div(100)
        options.box_threshold = box
        boxThresholdText.text = "Box Threshold: $box"
    }

    private fun updateMaskThreshold() {
        val mask = maskThresholdBar.progress.toFloat().div(100)
        options.mask_threshold = mask
        maskThresholdText.text = "Mask Threshold: $mask"
    }

    private fun setupSeekBarListener(seekBar: SeekBar, type: String) {
        seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if(type == "color") updateColorPreview()
                else if(type == "box") updateBoxThreshold()
                else if(type == "mask") updateMaskThreshold()
                else throw Exception("Invalid type: $type")
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {
            }
            override fun onStopTrackingTouch(seekBar: SeekBar?) {
            }
        })
    }
}
