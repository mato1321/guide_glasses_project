package com.rokid.facerecognition

import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.*

// ===== 資料模型（對應 Python API 的 JSON）=====

data class FaceResult(
    val name: String,
    val confidence: Double,
    val bbox: List<Double>
)

data class RecognizeResponse(
    val success: Boolean,
    val face_count: Int,
    val faces: List<FaceResult>
)

data class FaceListResponse(
    val faces: List<String>,
    val total: Int
)

data class StatusResponse(
    val status: String,
    val registered_faces: List<String>,
    val total_people: Int
)

// ===== API 介面 =====

interface FaceApiService {

    @GET("/")
    suspend fun getStatus(): Response<StatusResponse>

    @Multipart
    @POST("/recognize")
    suspend fun recognize(
        @Part file: MultipartBody.Part
    ): Response<RecognizeResponse>

    @GET("/faces")
    suspend fun listFaces(): Response<FaceListResponse>
}