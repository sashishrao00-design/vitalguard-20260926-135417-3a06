package ai.vitalguard

/**
 * Optional second screen: enter the non-optical vitals (SpO2 probe, cuff, thermometer)
 * and the patient MRN, then send them together with the PPG frames in one request so the
 * server can score risk in the same call. Keeping this in one round trip matters on
 * ward Wi-Fi.
 */
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class EntryActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { EntryScreen() }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EntryScreen() {
    var mrn by remember { mutableStateOf("") }
    var name by remember { mutableStateOf("") }
    var age by remember { mutableStateOf("") }
    var spo2 by remember { mutableStateOf("") }
    var rr by remember { mutableStateOf("") }
    var sbp by remember { mutableStateOf("") }
    var dbp by remember { mutableStateOf("") }
    var temp by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("idle") }

    // vitals travel with the PPG frames, so the server needs no second call
    val vitals: Map<String, Any?> = buildMap {
        spo2.toDoubleOrNull()?.let { put("spo2", it) }
        rr.toDoubleOrNull()?.let { put("rr", it) }
        sbp.toDoubleOrNull()?.let { put("sbp", it) }
        dbp.toDoubleOrNull()?.let { put("dbp", it) }
        temp.toDoubleOrNull()?.let { put("temp_c", it) }
        age.toDoubleOrNull()?.let { put("age_years", it) }
    }

    Scaffold(topBar = { CenterAlignedTopAppBar(title = { Text("VitalGuard AI · patient entry") }) }) { pad ->
        Column(Modifier.padding(pad).padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Field("MRN", mrn) { mrn = it }
            Field("Name", name) { name = it }
            Field("Age", age, KeyboardType.Number) { age = it }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Box(Modifier.weight(1f)) { Field("SpO2 %", spo2, KeyboardType.Decimal) { spo2 = it } }
                Box(Modifier.weight(1f)) { Field("Resp/min", rr, KeyboardType.Number) { rr = it } }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Box(Modifier.weight(1f)) { Field("Systolic", sbp, KeyboardType.Number) { sbp = it } }
                Box(Modifier.weight(1f)) { Field("Diastolic", dbp, KeyboardType.Number) { dbp = it } }
            }
            Field("Temp °C", temp, KeyboardType.Decimal) { temp = it }

            Button(onClick = {
                status = "registering…"
                Api.service.register("android", mapOf(
                    "mrn" to mrn.ifBlank { "MRN-UNKNOWN" }, "name" to name.ifBlank { mrn },
                    "age_years" to age.toDoubleOrNull(), "ward" to "self-screen", "unit" to "ward",
                )).enqueue(object : Callback<Map<String, Any>> {
                    override fun onResponse(c: Call<Map<String, Any>>, r: Response<Map<String, Any>>) {
                        status = "registered id=${r.body()?.get("id")} · ${vitals.size} vitals staged"
                    }
                    override fun onFailure(c: Call<Map<String, Any>>, t: Throwable) { status = "failed: ${t.message}" }
                })
            }, enabled = mrn.isNotBlank()) { Text("Register / update patient") }

            Text(status, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun Field(label: String, value: String, kbt: KeyboardType = KeyboardType.Text, onSet: (String) -> Unit) {
    OutlinedTextField(
        value = value, onValueChange = onSet, label = { Text(label) }, singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = kbt), modifier = Modifier.fillMaxWidth(),
    )
}
