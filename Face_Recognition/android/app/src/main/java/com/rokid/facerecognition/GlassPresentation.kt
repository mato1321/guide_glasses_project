package com.rokid.facerecognition

import android.app.Presentation
import android.content.Context
import android.os.Bundle
import android.view.Display
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView

/**
 * 投射到 Rokid 眼鏡鏡片上的畫面
 * 全螢幕 + 大字體 + 深色背景
 */
class GlassPresentation(
    context: Context,
    display: Display
) : Presentation(context, display) {

    private lateinit var resultContainer: LinearLayout
    private lateinit var scanningContainer: LinearLayout
    private lateinit var tvGlassIcon: TextView
    private lateinit var tvGlassName: TextView
    private lateinit var tvGlassConfidence: TextView
    private lateinit var tvGlassScanning: TextView
    private lateinit var tvGlassStatus: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.glass_overlay)

        // 綁定 UI
        resultContainer = findViewById(R.id.resultContainer)
        scanningContainer = findViewById(R.id.scanningContainer)
        tvGlassIcon = findViewById(R.id.tvGlassIcon)
        tvGlassName = findViewById(R.id.tvGlassName)
        tvGlassConfidence = findViewById(R.id.tvGlassConfidence)
        tvGlassScanning = findViewById(R.id.tvGlassScanning)
        tvGlassStatus = findViewById(R.id.tvGlassStatus)
    }

    /**
     * 顯示辨識結果（只顯示信心度最高的那個人）
     */
    fun showResult(name: String, confidence: Double) {
        scanningContainer.visibility = View.GONE
        resultContainer.visibility = View.VISIBLE

        if (name != "unknown") {
            tvGlassIcon.text = "✅"
            tvGlassName.text = name
            tvGlassName.setTextColor(0xFF00FF00.toInt())  // 綠色
            tvGlassConfidence.text = "信心度：${(confidence * 100).toInt()}%"
        } else {
            tvGlassIcon.text = "❓"
            tvGlassName.text = "未知人物"
            tvGlassName.setTextColor(0xFFFF4444.toInt())  // 紅色
            tvGlassConfidence.text = "信心度：${(confidence * 100).toInt()}%"
        }
    }

    /**
     * 顯示掃描中狀態
     */
    fun showScanning() {
        resultContainer.visibility = View.GONE
        scanningContainer.visibility = View.VISIBLE
        tvGlassScanning.text = "偵測中..."
    }

    /**
     * 更新底部狀態
     */
    fun updateStatus(status: String) {
        tvGlassStatus.text = status
    }
}