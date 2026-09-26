# VitalGuard AI — Android capture module (Member 1)

Kotlin/CameraX reference that produces **exactly the payload the browser page sends**
(`POST /api/ppg/process` with per-frame green-channel means). Keeping the payload
identical means the phone app can be dropped in without touching the Python side.

## Why stream numbers, not video
A 12 s 640×480 clip is ~30 MB over the wire; 240 frames of `(t, green_mean)` is ~6 KB.
The heavy lifting (filtering, peak detection, model) is on the server, so the phone only
needs the camera, the torch, and a mean per frame. `VitalGuardApi.kt` also includes the
`/api/ppg/video` multipart path if you prefer recording a clip instead.

> **Not compiled in this environment** — there is no Android SDK here, so these are
> complete, consistent reference sources that still need `Build` in Android Studio. The
> browser page at `/` exercises the identical server path today.

## Files here
| file | role |
|---|---|
| `CaptureActivity.kt` | CameraX + torch + ROI crop + frame sampling + POST |
| `MainActivity.kt` | Compose UI: patient field, other vitals, countdown, result |
| `VitalGuardApi.kt` | Retrofit service + payload models |
| `build.gradle.kts` | module config (minSdk 24, CameraX, OkHttp) |
| `AndroidManifest.xml` | camera / flash / internet permissions |

## Integrate
1. New Android Studio project, `app/src/main/java/ai/vitalguard/`.
2. Drop these four `.kt` files in, plus `AndroidManifest.xml` over the generated one.
3. Set `BASE_URL` in `VitalGuardApi.kt` to your machine's LAN IP (e.g. `http://192.168.1.20:8000/`).
   Android emulators reach the host as `http://10.0.2.2:8000/`.
4. Run. Point at a fingertip with the torch on, hold 12 s.
