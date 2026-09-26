package ai.vitalguard

/**
 * Step 1-2 of the build plan on the phone: open the rear camera with the torch,
 * preview the fingertip, sample the mean green value of a centred ROI per frame,
 * and stream the numbers to the FastAPI server.
 *
 * Why green: at ~540 nm oxy/deoxyhaemoglobin contrast is highest for superficial
 * capillary blood, so the pulse-to-noise ratio beats red on most phone sensors
 * (the sensor's own Bayer filter already weights green 2:1).
 */
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.util.Log
import android.util.Rational
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlin.math.min
import kotlin.math.roundToInt

class CaptureActivity : ComponentActivity() {

    private val required = arrayOf(Manifest.permission.CAMERA)
    private var analysis: ImageAnalysis? = null

    override fun onStart() {
        super.onStart()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) requestPermissions(required, 10) else bindCamera()
    }

    private fun bindCamera() {
        setContent { VitalGuardTheme { CaptureScreen(onStart = ::startCapture) } }
    }

    /**
     * Called by the UI button. Frames arrive on a background executor; we reduce
     * each one to a single float and buffer it. No pixels ever leave the phone.
     */
    private fun startCapture(durationMs: Long = 12_000, onDone: (List<FramePayload>) -> Unit) {
        val a = analysis ?: run { Log.e(TAG, "camera not bound yet"); return }
        val session = FrameSession(durationMs, onDone)
        a.setAnalyzer(ContextCompat.getMainExecutor(this)) { proxy ->
            if (session.closed) { proxy.close(); return@setAnalyzer }
            val img = proxy.image ?: run { proxy.close(); return@setAnalyzer }
            // YUV_420_888: the green plane is the second one and already 2x decimated
            val plane = img.planes[1]
            val buf = plane.buffer
            val rowStride = plane.rowStride
            val px = proxy.width / 2
            val py = proxy.height / 2
            val side = (min(px, py) * 0.30f).roundToInt()      // centred ROI
            if (side > 4) {
                val x0 = (px - side / 2).coerceAtLeast(0)
                val y0 = (py - side / 2).coerceAtLeast(0)
                var sum = 0L; var n = 0
                val step = if (side > 40) 2 else 1
                for (yy in 0 until side step step) {
                    val row = (y0 + yy) * rowStride + x0
                    for (xx in 0 until side step step) {
                        val i = row + xx * plane.pixelStride
                        if (i in 0 until buf.limit()) { sum += buf.get(i).toInt() and 0xFF; n++ }
                    }
                }
                if (n > 0) session.add(System.currentTimeMillis() - session.t0, sum.toDouble() / n)
            }
            proxy.close()
        }
        session.start()
    }

    /** Buffer + timing gate. Emits when enough frames were captured. */
    private class FrameSession(val durationMs: Long, val onDone: (List<FramePayload>) -> Unit) {
        val t0 = System.currentTimeMillis()
        private val out = ArrayList<FramePayload>(512)
        @Volatile var closed = false
        fun add(tMs: Long, greenMean: Double) {
            synchronized(out) {
                if (closed) return
                out.add(FramePayload(tMs / 1000.0, greenMean, greenMean))
                if (tMs >= durationMs) { closed = true; onDone(out.toList()) }
            }
        }
        fun start() { /* analyzer is already wired */ }
    }

    /**
     * Torch control. Call with true just before recording: this is what turns an
     * ambient-light video into a pseudo-Plethysmograph by transmitting through
     * the finger rather than reflecting off it.
     */
    fun setTorch(on: Boolean) {
        camera?.cameraControl?.enableTorch(on)
    }

    private var camera: androidx.camera.core.Camera? = null

    companion object { private const val TAG = "VitalGuardCapture" }
}

/** Compose screen: preview, ROI ring, countdown, result card. */
@Composable
fun CaptureScreen(onStart: (Long, (List<FramePayload>) -> Unit) -> Unit) {
    val preview = PreviewView(LocalContext.current).apply {
        scaleType = PreviewView.ScaleType.FILL_CENTER
    }
    var running by remember { mutableStateOf(false) }
    var progress by remember { mutableFloatStateOf(0f) }
    var result by remember { mutableStateOf<String?>(null) }

    Scaffold { pad ->
        Column(Modifier.padding(pad).fillMaxSize().background(Color(0xFF070C16))) {
            Box(Modifier.weight(1f).fillMaxWidth()) {
                AndroidView({ preview }, Modifier.fillMaxSize())
                Canvas(Modifier.fillMaxSize()) {
                    drawCircle(color = Color(0xFF5AD1A5), radius = size.minDimension * 0.17f,
                        center = Offset(size.width / 2, size.height * 0.5f), style = androidx.compose.ui.graphics.drawscope.Stroke(3f))
                }
                Surface(Modifier.align(Alignment.BottomCenter).padding(12.dp),
                    shape = RoundedCornerShape(12.dp), color = Color(0xCC0B1220)) {
                    Text(result ?: if (running) "hold still… ${(progress * 12).roundToInt()} s" else "cover the lens with your fingertip",
                        color = Color.White, style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp))
                }
            }
            Row(Modifier.fillMaxWidth().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button({
                    if (running) return@Button
                    running = true; progress = 0f
                    onStart(12_000) { frames ->
                        running = false
                        // POST /api/ppg/process  -> HR + risk + SHAP come back in one call
                        Api.service.process(ProcessRequest(frames, patientId = null, vitals = emptyMap(), save = false))
                            .enqueue(ResultHandler { result = it })
                    }
                }, Modifier.weight(1f)) { Text(if (running) "recording" else "capture 12 s") }
            }
        }
    }
}

private class ResultHandler(val onText: (String) -> Unit) : retrofit2.Callback<ProcessResponse> {
    override fun onResponse(c: retrofit2.Call<ProcessResponse>, r: retrofit2.Response<ProcessResponse>) {
        val b = r.body()
        onText(if (b == null || b.retryRequired() == true) b?.message ?: "no data"
               else "HR ${b.heart_rate.hr_bpm} BPM · risk ${b.risk?.risk_percent}% (${b.risk?.tier})")
    }
    override fun onFailure(c: retrofit2.Call<ProcessResponse>, t: Throwable) { onText("server unreachable: ${t.message}") }
    private fun ProcessResponse.retryRequired() = this.retry_required
}
