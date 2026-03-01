package com.rokid.facerecognition

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.display.DisplayManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.Display
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
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var previewView: PreviewView
    private lateinit var tvStatus: TextView
    private lateinit var tvGlassStatus: TextView
    private lateinit var tvResult: TextView
    private lateinit var btnCapture: Button
    private lateinit var btnConnect: Button
    private lateinit var switchAuto: Switch

    private lateinit var tts: TextToSpeech
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null

    // Rokid 眼鏡鏡片投射
    private var glassPresentation: GlassPresentation? = null
    private var glassDisplay: Display? = null

    // 持續偵測
    private val handler = Handler(Looper.getMainLooper())
    private var isAutoDetecting = false
    private val detectIntervalMs = 2000L  // 每 2 秒偵測一次

    // TTS 播報冷卻（10 秒）
    private var lastAnnounceName = ""
    private var lastAnnounceTime = 0L
    private val announceCooldown = 10000L  // 10 秒播報一次

    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 綁定 UI
        previewView = findViewById(R.id.previewView)
        tvStatus = findViewById(R.id.tvStatus)
        tvGlassStatus = findViewById(R.id.tvGlassStatus)
        tvResult = findViewById(R.id.tvResult)
        btnCapture = findViewById(R.id.btnCapture)
        btnConnect = findViewById(R.id.btnConnect)
        switchAuto = findViewById(R.id.switchAuto)

        // 初始化 TTS
        tts = TextToSpeech(this, this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 按鈕事件
        btnConnect.setOnClickListener { testConnection() }
        btnCapture.setOnClickListener { captureAndRecognize() }

        // 自動偵測開關
        switchAuto.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                startAutoDetection()
            } else {
                stopAutoDetection()
            }
        }

        // 偵測 Rokid 眼鏡（第二螢幕）
        setupGlassDisplay()

        // 監聽螢幕連接/斷開
        val displayManager = getSystemService(DISPLAY_SERVICE) as DisplayManager
        displayManager.registerDisplayListener(object : DisplayManager.DisplayListener {
            override fun onDisplayAdded(displayId: Int) {
                Log.d(TAG, "螢幕新增：$displayId")
                setupGlassDisplay()
            }
            override fun onDisplayRemoved(displayId: Int) {
                Log.d(TAG, "螢幕移除：$displayId")
                if (glassDisplay?.displayId == displayId) {
                    glassPresentation?.dismiss()
                    glassPresentation = null
                    glassDisplay = null
                    tvGlassStatus.text = "🔌 眼鏡已斷開"
                }
            }
            override fun onDisplayChanged(displayId: Int) {}
        }, handler)

        // 檢查相機權限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), 100
            )
        }
    }

    // ===== Rokid 眼鏡鏡片投射 =====

    private fun setupGlassDisplay() {
        val displayManager = getSystemService(DISPLAY_SERVICE) as DisplayManager
        val displays = displayManager.getDisplays(DisplayManager.DISPLAY_CATEGORY_PRESENTATION)

        if (displays.isNotEmpty()) {
            glassDisplay = displays[0]
            Log.d(TAG, "找到外部螢幕：${glassDisplay?.name}")
            tvGlassStatus.text = "🕶️ Rokid 眼鏡已連接（${glassDisplay?.name}）"

            // 建立並顯示 Presentation
            try {
                glassPresentation = GlassPresentation(this, glassDisplay!!)
                glassPresentation?.show()
                Log.d(TAG, "鏡片畫面已投射")
            } catch (e: Exception) {
                Log.e(TAG, "投射失敗：${e.message}")
                tvGlassStatus.text = "❌ 眼鏡投射失敗：${e.message}"
            }
        } else {
            tvGlassStatus.text = "📱 未偵測到 Rokid 眼鏡（僅手機模式）"
            Log.d(TAG, "沒有外部螢幕")
        }
    }

    /**
     * 將辨識結果投射到眼鏡鏡片
     * 只顯示信心度最高的那個人
     */
    private fun updateGlassDisplay(result: RecognizeResponse) {
        glassPresentation?.let { presentation ->
            if (result.faces.isEmpty()) {
                presentation.showScanning()
                return
            }

            // 找出信心度最高的人臉
            val bestFace = result.faces.maxByOrNull { it.confidence } ?: return
            presentation.showResult(bestFace.name, bestFace.confidence)
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
        glassPresentation?.showScanning()
    }

    // ===== 相機 =====

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.getSurfaceProvider())
            }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )
                tvStatus.text = "📷 相機已啟動"
            } catch (e: Exception) {
                tvStatus.text = "❌ 相機啟動失敗：${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ===== 拍照並辨識 =====

    private fun captureAndRecognize() {
        val imageCapture = imageCapture ?: return

        if (!isAutoDetecting) {
            tvResult.text = "⏳ 辨識中..."
            btnCapture.isEnabled = false
        }

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    val bitmap = imageProxyToBitmap(imageProxy)
                    imageProxy.close()

                    if (bitmap != null) {
                        sendToServer(bitmap)
                    } else {
                        if (!isAutoDetecting) {
                            tvResult.text = "❌ 圖片轉換失敗"
                            btnCapture.isEnabled = true
                        }
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    if (!isAutoDetecting) {
                        tvResult.text = "❌ 拍照失敗：${exception.message}"
                        btnCapture.isEnabled = true
                    }
                }
            }
        )
    }

    // ===== 送到 Python 伺服器 =====

    private fun sendToServer(bitmap: Bitmap) {
        lifecycleScope.launch {
            try {
                val stream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 85, stream)
                val bytes = stream.toByteArray()

                val requestBody = bytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                val part = MultipartBody.Part.createFormData("file", "capture.jpg", requestBody)

                val response = withContext(Dispatchers.IO) {
                    ApiClient.faceApi.recognize(part)
                }

                if (response.isSuccessful) {
                    val result = response.body()!!
                    displayResult(result)
                    // 投射到眼鏡鏡片
                    updateGlassDisplay(result)
                } else {
                    if (!isAutoDetecting) {
                        tvResult.text = "❌ 伺服器錯誤：${response.code()}"
                    }
                }
            } catch (e: Exception) {
                if (!isAutoDetecting) {
                    tvResult.text = "❌ 連線失敗：${e.message}"
                }
            } finally {
                if (!isAutoDetecting) {
                    btnCapture.isEnabled = true
                }
            }
        }
    }

    // ===== 顯示辨識結果 + TTS（10 秒冷卻）=====

    private fun displayResult(result: RecognizeResponse) {
        if (result.faces.isEmpty()) {
            tvResult.text = if (isAutoDetecting) "🔄 偵測中...（未發現人臉）" else "❌ 未偵測到人臉"
            return
        }

        // 找出信心度最高的人臉
        val bestFace = result.faces.maxByOrNull { it.confidence } ?: return

        // 手機上顯示結果
        val sb = StringBuilder()
        sb.appendLine("偵測到 ${result.face_count} 張人臉，最佳匹配：")

        if (bestFace.name != "unknown") {
            sb.appendLine("✅ ${bestFace.name}（${(bestFace.confidence * 100).toInt()}%）")

            // TTS 語音播報（每 10 秒一次）
            val now = System.currentTimeMillis()
            if (bestFace.name != lastAnnounceName || now - lastAnnounceTime > announceCooldown) {
                lastAnnounceName = bestFace.name
                lastAnnounceTime = now
                tts.speak(
                    "前方是${bestFace.name}",
                    TextToSpeech.QUEUE_FLUSH,
                    null,
                    bestFace.name
                )
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
                val response = withContext(Dispatchers.IO) {
                    ApiClient.faceApi.getStatus()
                }
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

    // ===== 工具函數 =====

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val image = imageProxy.image ?: return null
            val planes = image.planes

            if (imageProxy.format == ImageFormat.JPEG) {
                val buffer = planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            } else {
                val yBuffer = planes[0].buffer
                val uBuffer = planes[1].buffer
                val vBuffer = planes[2].buffer

                val ySize = yBuffer.remaining()
                val uSize = uBuffer.remaining()
                val vSize = vBuffer.remaining()

                val nv21 = ByteArray(ySize + uSize + vSize)
                yBuffer.get(nv21, 0, ySize)
                vBuffer.get(nv21, ySize, vSize)
                uBuffer.get(nv21, ySize + vSize, uSize)

                val yuvImage = YuvImage(
                    nv21, ImageFormat.NV21,
                    imageProxy.width, imageProxy.height, null
                )
                val out = ByteArrayOutputStream()
                yuvImage.compressToJpeg(
                    Rect(0, 0, imageProxy.width, imageProxy.height),
                    90, out
                )
                val jpegBytes = out.toByteArray()
                BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.TRADITIONAL_CHINESE
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            Toast.makeText(this, "需要相機權限才能使用", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopAutoDetection()
        cameraExecutor.shutdown()
        glassPresentation?.dismiss()
        tts.shutdown()
    }
}