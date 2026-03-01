package com.rokid.facerecognition

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object ApiClient {

    // ⚠️ 改成你電腦的區域網路 IP
    // 在你的電腦 CMD 輸入 ipconfig，找到 IPv4 位址
    // 例如：192.168.1.100
    // 如果用模擬器測試，用 10.0.2.2（模擬器專用，代表電腦 localhost）
    private const val BASE_URL = "http://192.168.0.112:8000"

    private val logging = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val client = OkHttpClient.Builder()
        .addInterceptor(logging)
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    val faceApi: FaceApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(FaceApiService::class.java)
    }
}