package com.rokid.facerecognition

import android.media.AudioAttributes
import android.media.MediaPlayer
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Log
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
    private lateinit var tvResult: TextView

    private lateinit var tts: TextToSpeech
    private lateinit var cameraExecutor: ExecutorService

    // 自動偵測（每 10 秒）
    private val handler = Handler(Looper.getMainLooper())
    private val detectIntervalMs = 10000L

    // 防止同時送多個請求
    private val isProcessing = AtomicBoolean(false)

    // 最新的 JPEG bytes
    private var latestJpegBytes: ByteArray? = null
    private val jpegLock = Object()

    // TTS 播報冷卻（10 秒）
    private var lastAnnounceName = ""
    private var lastAnnounceTime = 0L
    private val announceCooldown = 10000L

    // TTS 是否可用
    private var ttsReady = false

    private val TAG = "FaceRecognition"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 全螢幕沉浸模式
        window.decorView.systemUiVisibility = (
                android.view.View.SYSTEM_UI_FLAG_FULLSCREEN
                        or android.view.View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        or android.view.View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                )

        previewView = findViewById(R.id.previewView)
        tvResult = findViewById(R.id.tvResult)

        tts = TextToSpeech(this, this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 檢查相機權限
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

    // ===== 相機 =====

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .build()
                .also {
                    it.setSurfaceProvider(previewView.getSurfaceProvider())
                }

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
                // 相機啟動成功，開始自動偵測
                startAutoDetection()
            } catch (e: Exception) {
                tvResult.text = "相機啟動失敗"
                Log.e(TAG, "相機失敗", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ===== RGBA → JPEG =====

    private fun rgbaToJpeg(imageProxy: ImageProxy): ByteArray? {
        return try {
            val plane = imageProxy.planes[0]
            val buffer: ByteBuffer = plane.buffer
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val width = imageProxy.width
            val height = imageProxy.height

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

            if (rowStride == width * pixelStride) {
                buffer.rewind()
                bitmap.copyPixelsFromBuffer(buffer)
            } else {
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

    // ===== 自動偵測（App 啟動就開始，每 10 秒一次）=====

    private val autoDetectRunnable = object : Runnable {
        override fun run() {
            captureAndRecognize()
            handler.postDelayed(this, detectIntervalMs)
        }
    }

    private fun startAutoDetection() {
        tvResult.text = "偵測中..."
        handler.post(autoDetectRunnable)
    }

    private fun stopAutoDetection() {
        handler.removeCallbacks(autoDetectRunnable)
    }

    // ===== 擷取並辨識 =====

    private fun captureAndRecognize() {
        if (!isProcessing.compareAndSet(false, true)) return

        val jpegBytes: ByteArray?
        synchronized(jpegLock) {
            jpegBytes = latestJpegBytes?.clone()
        }

        if (jpegBytes == null) {
            isProcessing.set(false)
            return
        }

        sendToServer(jpegBytes)
    }

    // ===== 送到伺服器 =====

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
                }
            } catch (e: Exception) {
                Log.e(TAG, "連線失敗", e)
            } finally {
                isProcessing.set(false)
            }
        }
    }

    // ===== 顯示結果 + 語音播報 =====

    private fun displayResult(result: RecognizeResponse) {
        if (result.faces.isEmpty()) {
            tvResult.text = "未偵測到人臉"
            return
        }

        // 只取信心度最高的
        val bestFace = result.faces.maxByOrNull { it.confidence } ?: return

        if (bestFace.name != "unknown") {
            val confidence = (bestFace.confidence * 100).toInt()
            tvResult.text = "${bestFace.name}  ${confidence}%"

            // 語音播報（每 10 秒一次）
            val now = System.currentTimeMillis()
            if (bestFace.name != lastAnnounceName || now - lastAnnounceTime > announceCooldown) {
                lastAnnounceName = bestFace.name
                lastAnnounceTime = now

                val message = "你面前的人是${bestFace.name}"
                Log.d(TAG, "🔊 播報：$message")

                if (ttsReady) {
                    val speakResult = tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, bestFace.name)
                    if (speakResult == TextToSpeech.ERROR) {
                        speakWithMediaPlayer(message)
                    }
                } else {
                    speakWithMediaPlayer(message)
                }
            }
        } else {
            tvResult.text = "未偵測到人臉"
        }
    }

    // ===== TTS 初始化 =====

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            var result = tts.setLanguage(Locale.TRADITIONAL_CHINESE)

            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                result = tts.setLanguage(Locale.SIMPLIFIED_CHINESE)
            }
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                result = tts.setLanguage(Locale.US)
            }

            if (result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED) {
                ttsReady = true
                tts.setSpeechRate(1.0f)
                tts.setPitch(1.0f)
                Log.d(TAG, "✅ TTS 初始化成功")
            } else {
                Log.e(TAG, "❌ TTS 沒有可用語言")
            }
        } else {
            Log.e(TAG, "❌ TTS 初始化失敗")
        }
    }

    // ===== 備用語音播報 =====

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
        } catch (e: Exception) {
            Log.e(TAG, "MediaPlayer 播報失敗：${e.message}")
        }
    }

    // ===== 權限 =====

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            Toast.makeText(this, "需要相機權限", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopAutoDetection()
        cameraExecutor.shutdown()
        tts.shutdown()
    }
}