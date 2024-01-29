package ai.onnxruntime.example.roofSegmenter

import android.content.Context
import android.content.SharedPreferences

class Options(context: Context) {
    private val preferences: SharedPreferences = context.getSharedPreferences("Options", Context.MODE_PRIVATE)

    private val MERGE_MASKS = "merge_masks"
    private val COLOR = "color"

    var merge_masks: Boolean
        get() = preferences.getBoolean(MERGE_MASKS, false)
        set(value) = preferences.edit().putBoolean(MERGE_MASKS, value).apply()

    var color: Int
        get() = preferences.getInt(COLOR, 0)
        set(value) = preferences.edit().putInt(COLOR, value).apply()
}