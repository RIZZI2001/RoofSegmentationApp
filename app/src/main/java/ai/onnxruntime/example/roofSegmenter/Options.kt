package ai.onnxruntime.example.roofSegmenter

import android.content.Context
import android.content.SharedPreferences

class Options(context: Context) {
    private val preferences: SharedPreferences = context.getSharedPreferences("Options", Context.MODE_PRIVATE)

    private val MERGE_MASKS = "merge_masks"
    private val COLOR = "color"
    private val BOX_THRESHOLD = "box_threshold"
    private val MASK_THRESHOLD = "mask_threshold"

    var merge_masks: Boolean
        get() = preferences.getBoolean(MERGE_MASKS, false)
        set(value) = preferences.edit().putBoolean(MERGE_MASKS, value).apply()

    var color: Int
        get() = preferences.getInt(COLOR, 0)
        set(value) = preferences.edit().putInt(COLOR, value).apply()

    var box_threshold: Float
        get() = preferences.getFloat(BOX_THRESHOLD, 0.5f)
        set(value) = preferences.edit().putFloat(BOX_THRESHOLD, value).apply()

    var mask_threshold: Float
        get() = preferences.getFloat(MASK_THRESHOLD, 0.7f)
        set(value) = preferences.edit().putFloat(MASK_THRESHOLD, value).apply()
}