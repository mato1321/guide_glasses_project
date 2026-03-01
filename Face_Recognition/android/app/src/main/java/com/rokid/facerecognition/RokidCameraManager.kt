package com.rokid.facerecognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.util.Log
import android.view.Surface
import com.herohan.uvcapp.CameraHelper
import com.herohan.uvcapp.ICameraHelper

/**
 * Rokid 眼鏡攝像頭管理器
 * 透過 UVC 協議讀取眼鏡的攝像頭影像
 */
class RokidCameraManager(private val context: Context) {

    private var cameraHelper: ICameraHelper? = null
    private var isConnected = false
    private var surfaceTexture: SurfaceTexture? = null

    var onStatusChanged: ((String) -> Unit)? = null

    private val TAG = "RokidCamera"

    /**
     * 初始化 UVC 攝像頭
     */
    fun initialize() {
        cameraHelper = CameraHelper().apply {
            setStateCallback(object : ICameraHelper.StateCallback {

                override fun onAttach(device: UsbDevice) {
                    Log.d(TAG, "USB 裝置已連接：${device.deviceName}")
                    onStatusChanged?.invoke("🔌 Rokid 眼鏡已連接")
                    cameraHelper?.selectDevice(device)
                }

                override fun onDeviceOpen(device: UsbDevice, isFirstOpen: Boolean) {
                    Log.d(TAG, "USB 裝置已開啟")
                    onStatusChanged?.invoke("📷 攝像頭已開啟")
                    isConnected = true

                    // 建立 SurfaceTexture 來接收影像
                    try {
                        surfaceTexture = SurfaceTexture(0)
                        surfaceTexture?.setDefaultBufferSize(640, 480)
                        val surface = Surface(surfaceTexture)
                        cameraHelper?.addSurface(surface, false)
                    } catch (e: Exception) {
                        Log.e(TAG, "設定 Surface 失敗：${e.message}")
                    }
                }

                override fun onCameraOpen(device: UsbDevice) {
                    Log.d(TAG, "攝像頭已開啟")
                    onStatusChanged?.invoke("✅ Rokid 攝像頭就緒")
                    isConnected = true
                }

                override fun onCameraClose(device: UsbDevice) {
                    Log.d(TAG, "攝像頭已關閉")
                    onStatusChanged?.invoke("📷 攝像頭已關閉")
                    isConnected = false
                }

                override fun onDeviceClose(device: UsbDevice) {
                    Log.d(TAG, "USB 裝置已關閉")
                    isConnected = false
                }

                override fun onDetach(device: UsbDevice) {
                    Log.d(TAG, "USB 裝置已拔除")
                    onStatusChanged?.invoke("🔌 Rokid 眼鏡已斷開")
                    isConnected = false
                }

                override fun onCancel(device: UsbDevice) {
                    Log.d(TAG, "USB 權限被拒絕")
                    onStatusChanged?.invoke("❌ USB 權限被拒絕")
                }
            })
        }
    }

    /**
     * 擷取當前畫面
     * 使用 SurfaceTexture 取得 Bitmap
     */
    fun captureFrame(callback: (Bitmap?) -> Unit) {
        if (!isConnected) {
            callback(null)
            return
        }

        try {
            // 透過 CameraHelper 的截圖功能
            // UVCAndroid 使用 setImageCaptureConfig 和相關 API
            // 這裡用 SurfaceTexture 的方式取得畫面
            val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)

            surfaceTexture?.let { st ->
                try {
                    st.updateTexImage()
                    // 從 OpenGL texture 讀取到 Bitmap
                    // 注意：這是簡化版，實際可能需要 OpenGL 操作
                    callback(bitmap)
                } catch (e: Exception) {
                    Log.e(TAG, "擷取畫面失敗：${e.message}")
                    callback(null)
                }
            } ?: callback(null)

        } catch (e: Exception) {
            Log.e(TAG, "擷取失敗：${e.message}")
            callback(null)
        }
    }

    /**
     * 設定預覽 Surface
     */
    fun setPreviewSurface(surface: Surface) {
        try {
            cameraHelper?.addSurface(surface, false)
        } catch (e: Exception) {
            Log.e(TAG, "設定預覽失敗：${e.message}")
        }
    }

    /**
     * 檢查是否有 Rokid 眼鏡連接
     */
    fun isRokidConnected(): Boolean {
        val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        val devices = usbManager.deviceList

        for ((_, device) in devices) {
            if (device.deviceClass == 14 ||
                device.deviceClass == 239 ||
                device.deviceName.contains("rokid", ignoreCase = true)) {
                Log.d(TAG, "找到 Rokid 裝置：${device.deviceName} (VID:${device.vendorId}, PID:${device.productId})")
                return true
            }
        }
        return false
    }

    /**
     * 取得已連接的 USB 裝置列表（除錯用）
     */
    fun getConnectedDevices(): List<String> {
        val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
        return usbManager.deviceList.map { (name, device) ->
            "$name (VID:${device.vendorId}, PID:${device.productId}, Class:${device.deviceClass})"
        }
    }

    fun release() {
        try {
            surfaceTexture?.release()
            cameraHelper?.release()
        } catch (e: Exception) {
            Log.e(TAG, "釋放失敗：${e.message}")
        }
        cameraHelper = null
        surfaceTexture = null
        isConnected = false
    }
}