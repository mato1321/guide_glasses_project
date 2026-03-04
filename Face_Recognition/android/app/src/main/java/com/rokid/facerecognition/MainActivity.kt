package com.rokid.facerecognition

import android.media.AudioAttributes
import android.media.MediaPlayer
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Button
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var tvStatus: TextView
    private lateinit var tvResult: TextView
    private lateinit var btnCapture: Button
    private lateinit var btnConnect: Button
    private lateinit var switchAuto: Switch

    private lateinit var tts: TextToSpeech
    private lateinit var cameraExecutor: ExecutorService

    // 持續偵測
    private val handler = Handler(Looper.getMainLooper())
    private var isAutoDetecting = false
    private val detectIntervalMs = 3000L

    // 防止同時送多個請求
    private val isProcessing = AtomicBoolean(false)

    // 最新的 JPEG bytes（直接存 JPEG，不做 YUV 轉換）
    private var latestJpegBytes: ByteArray? = null
    private val jpegLock = Object()

    // TTS 播報冷卻（10 秒）
    private var lastAnnounceName = ""
    private var lastAnnounceTime = 0L
    private val announceCooldown = 10000L

    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        tvStatus = findViewById(R.id.tvStatus)
        tvResult = findViewById(R.id.tvResult)
        btnCapture = findViewById(R.id.btnCapture)
        btnConnect = findViewById(R.id.btnConnect)
        switchAuto = findViewById(R.id.switchAuto)

        tts = TextToSpeech(this, this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        btnConnect.setOnClickListener {
            // 測試語音
            tts.speak("測試語音", TextToSpeech.QUEUE_FLUSH, null, "test")
            speakWithMediaPlayer("測試語音")
            // 原本的連線測試
            testConnection()
        }
        btnCapture.setOnClickListener { captureAndRecognize() }

        switchAuto.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) startAutoDetection() else stopAutoDetection()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), 100
            )
        }
    }

    // ===== 相機：用 JPEG 格式直接輸出，避免 YUV 問題 =====

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // 預覽
            val preview = Preview.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .build()
                .also {
                    it.setSurfaceProvider(previewView.getSurfaceProvider())
                }

            // 影像分析：強制 RGBA 輸出（不用 YUV）
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val bytes = rgbaToJpeg(imageProxy)
                    if (bytes != null) {
                        synchronized(jpegLock) {
                            latestJpegBytes = bytes
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "影像轉換失敗：${e.message}")
                } finally {
                    imageProxy.close()
                }
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )
                tvStatus.text = "📷 相機已啟動"
            } catch (e: Exception) {
                tvStatus.text = "❌ 相機啟動失敗：${e.message}"
                Log.e(TAG, "相機失敗", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ===== RGBA → JPEG（安全轉換，不依賴 YUV）=====

    private fun rgbaToJpeg(imageProxy: ImageProxy): ByteArray? {
        return try {
            val plane = imageProxy.planes[0]
            val buffer: ByteBuffer = plane.buffer
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val width = imageProxy.width
            val height = imageProxy.height

            // 建立 Bitmap
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

            // 處理 row padding
            if (rowStride == width * pixelStride) {
                // 沒有 padding，直接複製
                buffer.rewind()
                bitmap.copyPixelsFromBuffer(buffer)
            } else {
                // 有 padding，逐行複製
                val rowBytes = width * pixelStride
                val rowBuffer = ByteArray(rowStride)
                val pixels = IntArray(width)

                for (y in 0 until height) {
                    buffer.position(y * rowStride)
                    buffer.get(rowBuffer, 0, minOf(rowStride, buffer.remaining()))

                    for (x in 0 until width) {
                        val offset = x * pixelStride
                        val r = rowBuffer[offset].toInt() and 0xFF
                        val g = rowBuffer[offset + 1].toInt() and 0xFF
                        val b = rowBuffer[offset + 2].toInt() and 0xFF
                        val a = rowBuffer[offset + 3].toInt() and 0xFF
                        pixels[x] = (a shl 24) or (r shl 16) or (g shl 8) or b
                    }
                    bitmap.setPixels(pixels, 0, width, 0, y, width, 1)
                }
            }

            // 處理旋轉
            val rotation = imageProxy.imageInfo.rotationDegrees
            val finalBitmap = if (rotation != 0) {
                val matrix = Matrix()
                matrix.postRotate(rotation.toFloat())
                val rotated = Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                )
                bitmap.recycle()
                rotated
            } else {
                bitmap
            }

            // Bitmap → JPEG bytes
            val stream = ByteArrayOutputStream()
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 75, stream)
            finalBitmap.recycle()
            val jpegBytes = stream.toByteArray()
            stream.close()

            jpegBytes
        } catch (e: Exception) {
            Log.e(TAG, "RGBA→JPEG 失敗：${e.message}")
            null
        }
    }

    // ===== 持續自動偵測 =====

    private val autoDetectRunnable = object : Runnable {
        override fun run() {
            if (isAutoDetecting) {
                captureAndRecognize()
                handler.postDelayed(this, detectIntervalMs)
            }
        }
    }

    private fun startAutoDetection() {
        isAutoDetecting = true
        btnCapture.isEnabled = false
        btnCapture.text = "🔄 自動偵測中..."
        tvResult.text = "🔄 自動偵測已開啟（每 ${detectIntervalMs / 1000} 秒）"
        handler.post(autoDetectRunnable)
    }

    private fun stopAutoDetection() {
        isAutoDetecting = false
        handler.removeCallbacks(autoDetectRunnable)
        btnCapture.isEnabled = true
        btnCapture.text = "📷 拍照辨識"
        tvResult.text = "自動偵測已關閉"
    }

    // ===== 擷取並辨識 =====

    private fun captureAndRecognize() {
        if (!isProcessing.compareAndSet(false, true)) return

        val jpegBytes: ByteArray?
        synchronized(jpegLock) {
            jpegBytes = latestJpegBytes?.clone()
        }

        if (jpegBytes == null) {
            if (!isAutoDetecting) tvResult.text = "❌ 尚未取得影像，請稍候"
            isProcessing.set(false)
            return
        }

        if (!isAutoDetecting) {
            tvResult.text = "⏳ 辨識中..."
            btnCapture.isEnabled = false
        }

        sendToServer(jpegBytes)
    }

    // ===== 送到伺服器（直接送 JPEG bytes，不再轉換）=====

    private fun sendToServer(jpegBytes: ByteArray) {
        lifecycleScope.launch {
            try {
                val requestBody = jpegBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                val part = MultipartBody.Part.createFormData("file", "capture.jpg", requestBody)

                val response = withContext(Dispatchers.IO) {
                    ApiClient.faceApi.recognize(part)
                }

                if (response.isSuccessful) {
                    val result = response.body()!!
                    displayResult(result)
                } else {
                    if (!isAutoDetecting) tvResult.text = "❌ 伺服器錯誤：${response.code()}"
                }
            } catch (e: Exception) {
                if (!isAutoDetecting) tvResult.text = "❌ 連線失敗：${e.message}"
                Log.e(TAG, "送出失敗", e)
            } finally {
                isProcessing.set(false)
                if (!isAutoDetecting) btnCapture.isEnabled = true
            }
        }
    }

    // ===== 顯示結果 + TTS =====

    private fun displayResult(result: RecognizeResponse) {
        if (result.faces.isEmpty()) {
            tvResult.text = if (isAutoDetecting) "🔄 偵測中...（未發現人臉）" else "❌ 未偵測到人臉"
            return
        }

        val bestFace = result.faces.maxByOrNull { it.confidence } ?: return
        val sb = StringBuilder()
        sb.appendLine("偵測到 ${result.face_count} 張人臉，最佳匹配：")

        if (bestFace.name != "unknown") {
            sb.appendLine("✅ ${bestFace.name}（${(bestFace.confidence * 100).toInt()}%）")

            val now = System.currentTimeMillis()
            if (bestFace.name != lastAnnounceName || now - lastAnnounceTime > announceCooldown) {
                lastAnnounceName = bestFace.name
                lastAnnounceTime = now

                val message = "前方是${bestFace.name}"
                Log.d(TAG, "🔊 嘗試 TTS 播報：$message")

                // 方法 1：用 TTS 引擎
                val speakResult = tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, bestFace.name)

                if (speakResult == TextToSpeech.ERROR) {
                    Log.e(TAG, "❌ TTS 失敗，改用 MediaPlayer")
                    // 方法 2：備用 MediaPlayer
                    speakWithMediaPlayer(message)
                }
            }
        } else {
            sb.appendLine("❓ 未知人物（${(bestFace.confidence * 100).toInt()}%）")
        }

        tvResult.text = sb.toString().trim()
    }

    // ===== 連線測試 =====

    private fun testConnection() {
        tvStatus.text = "🔄 連線中..."
        lifecycleScope.launch {
            try {
                val response = withContext(Dispatchers.IO) { ApiClient.faceApi.getStatus() }
                if (response.isSuccessful) {
                    val status = response.body()!!
                    tvStatus.text = "✅ 已連線！已註冊 ${status.total_people} 人：${status.registered_faces}"
                } else {
                    tvStatus.text = "❌ 伺服器錯誤：${response.code()}"
                }
            } catch (e: Exception) {
                tvStatus.text = "❌ 連線失敗：${e.message}"
            }
        }
    }

    private fun speakWithMediaPlayer(text: String) {
        try {
            val url = "https://translate.google.com/translate_tts?ie=UTF-8&tl=zh-TW&client=tw-ob&q=${
                java.net.URLEncoder.encode(text, "UTF-8")
            }"

            val mediaPlayer = MediaPlayer()
            mediaPlayer.setAudioAttributes(
                AudioAttributes.Builder()
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .build()
            )
            mediaPlayer.setDataSource(url)
            mediaPlayer.setOnPreparedListener { it.start() }
            mediaPlayer.setOnCompletionListener { it.release() }
            mediaPlayer.prepareAsync()
            Log.d(TAG, "🔊 MediaPlayer 備用播報：$text")
        } catch (e: Exception) {
            Log.e(TAG, "❌ MediaPlayer 播報失敗：${e.message}")
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            // 先嘗試繁體中文
            var result = tts.setLanguage(Locale.TRADITIONAL_CHINESE)

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                // 再嘗試簡體中文
                result = tts.setLanguage(Locale.SIMPLIFIED_CHINESE)
            }

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                // 再嘗試英文
                result = tts.setLanguage(Locale.US)
            }

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(TAG, "❌ TTS：沒有任何語言可用")
                runOnUiThread {
                    tvStatus.text = tvStatus.text.toString() + "\n⚠️ 語音引擎不可用"
                }
            } else {
                // 設定語速和音量
                tts.setSpeechRate(1.0f)
                tts.setPitch(1.0f)
                Log.d(TAG, "✅ TTS 初始化成功，語言：${tts.voice?.locale}")

                // 測試播報一句
                tts.speak("語音系統已就緒", TextToSpeech.QUEUE_FLUSH, null, "init")
            }
        } else {
            Log.e(TAG, "❌ TTS 初始化失敗，status=$status")
            runOnUiThread {
                tvStatus.text = tvStatus.text.toString() + "\n❌ 語音引擎初始化失敗"
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            Toast.makeText(this, "需要相機權限才能使用", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopAutoDetection()
        cameraExecutor.shutdown()
        tts.shutdown()
    }
}