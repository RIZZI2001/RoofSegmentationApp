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
        setupSeekBarListener(seekBarRed)
        setupSeekBarListener(seekBarGreen)
        setupSeekBarListener(seekBarBlue)
        setupSeekBarListener(seekBarAlpha)
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

    private fun setupSeekBarListener(seekBar: SeekBar) {
        seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                updateColorPreview()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {
            }
            override fun onStopTrackingTouch(seekBar: SeekBar?) {
            }
        })
    }
}
