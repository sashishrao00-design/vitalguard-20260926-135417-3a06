package ai.vitalguard

import com.google.gson.Gson
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.Call
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.Header
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Query
import java.io.File
import java.util.concurrent.TimeUnit

/** Base URL of the FastAPI server. LAN IP of the laptop running `python -m vitalguard.server`. */
const val BASE_URL = "http://10.0.2.2:8000/"     // emulator -> host; use http://192.168.x.x:8000/ on a device

data class FramePayload(val t: Double, val v: Double, val luma: Double)

data class ProcessRequest(
    val frames: List<FramePayload>,
    @com.google.gson.annotations.SerializedName("patient_id") val patientId: Int?,
    val vitals: Map<String, Any?>,
    val save: Boolean = true,
)

/** Mirrors vitalguard/server.py -> /api/ppg/process response (only fields we render). */
data class HeartRateResult(
    val hr_bpm: Double?,
    val quality: Double,
    val quality_label: String,
    val hrv_rmssd_ms: Double?,
    val breathing_rate_bpm: Double?,
    val guidance: List<String>?,
)
data class RiskResult(
    val risk_percent: Double,
    val tier: String,
    val news2: Double?,
    val top_factors: List<Factor>?,
    val advice: Advice?,
)
data class Factor(val label: String, val value_text: String, val shap: Double, val text: String)
data class Advice(val action: String, val detail: String, val recheck_min: Int)
data class ProcessResponse(
    val heart_rate: HeartRateResult,
    val risk: RiskResult?,
    val retry_required: Boolean?,
    val message: String?,
)

interface VitalGuardService {
    @POST("api/ppg/process")
    fun process(@Body body: ProcessRequest): Call<ProcessResponse>

    /** Fallback path: upload the raw recorded clip and let OpenCV decode it. */
    @Multipart
    @POST("api/ppg/video")
    fun uploadVideo(
        @Part clip: MultipartBody.Part,
        @Query("patient_id") patientId: Int?,
        @Part("vitals") vitals: MultipartBody.Part?,
    ): Call<ProcessResponse>

    @POST("api/patients")
    fun register(@Header("X-Device") device: String, @Body body: Map<String, Any?>): Call<Map<String, Any>>
}

object Api {
    val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(45, TimeUnit.SECONDS)
            .addInterceptor(HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BASIC })
            .build()
    }
    val service: VitalGuardService by lazy {
        Retrofit.Builder().baseUrl(BASE_URL).client(client)
            .addConverterFactory(GsonConverterFactory.create()).build()
            .create(VitalGuardService::class.java)
    }
    val gson = Gson()

    fun videoPart(file: File) = MultipartBody.Part.createFormData("file", file.name,
        file.asRequestBody("video/*".toMediaType()))
}
