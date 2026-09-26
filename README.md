# VitalGuard AI — working prototype

> **Detect deterioration before it becomes critical.**

Phone camera + flash → fingertip → PPG → heart rate → AI risk score → SHAP explanation → alert → doctor dashboard.

This repo is the runnable prototype for all 10 steps of the build plan. Everything
marked ✅ runs today; the two items that need physical hardware are marked 🔌 with the
code ready for them.

---

## Hosting reliability — important

Repeated Cloudflare 1033 failures occurred even after adding the watchdog. It can
create a *new* working URL, but cannot repair the old link already sent to a phone.
Neither a successful health check nor SSH keepalives guarantee future availability.
The precise cause of the repeated connection losses is unconfirmed.

An alternative temporary tunnel, tested with the browser and API suites:

```bash
ssh -T -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
  -o ExitOnForwardFailure=yes -R 80:localhost:8000 nokey@localhost.run
```

Keep the API and SSH process running. Use the HTTPS address it prints. Free
localhost.run domains also change; a local SSH key alone does **not** reserve one.
For a stable demo address, deploy the app on a hosting account, or configure a
provider-managed custom domain. This requires the owner's account/setup and has
not been done here. Public prototype links have no authentication: use fictional
data only.

## Open it on your phone

```bash
./phone_link.sh watch     # public https URL + keeps itself alive; writes web/phone-qr.png
```

It starts a Cloudflare quick tunnel to `:8000`, waits until the edge actually answers
`200`, regenerates the QR so it encodes whatever URL it got (nothing is hardcoded), then
**supervises**: every 25 s it fetches `<url>/api/health` from the public side and, if the
tunnel stops serving, republishes and prints the new URL. Modes: `start` (one-shot),
`watch` (recommended for a demo), `status` (diagnose), `stop`.

| URL | what |
|---|---|
| `<tunnel>/#capture` | point the camera at your fingertip, get HR + risk + SHAP |
| `<tunnel>/phone` | landing page with the QR + instructions, so the whole team can join |
| `<tunnel>/#ward` | doctor dashboard |
| `<tunnel>/docs` | Swagger UI — useful for poking the API from a laptop |

**Why it works over the internet:** camera access needs a *secure context*, and the tunnel
terminates TLS, so `getUserMedia` and the torch control are permitted on Android Chrome.
iOS Safari can capture but exposes no torch API, so the flash button stays disabled there.

### "Error 1033" / "Cloudflare 530" — what it means and what to do

If the phone shows Cloudflare **error 1033**, that is HTTP **530**: the hostname exists but
no `cloudflared` is connected to it any more. It is *not* a bug in the prototype — the app can
be perfectly healthy on `:8000` while the public URL is dead, and nothing client-side explains
it. Quick tunnels die on their own; the observed failure was a QUIC idle timeout
(`timeout: no recent network activity`) where cloudflared re-registered but the hostname stayed
530 forever, with the process still looking alive.

Three deliberate defences:

1. **`--protocol http2`** instead of QUIC/UDP: the observed death was a QUIC idle timeout, and
   this sandbox could not raise the UDP receive buffer cloudflared asks for (it warns on start).
2. **`watch` mode** probes the public URL and republishes (~35 s measured), so a dropped
   tunnel is a hiccup rather than a dead demo.
3. **`./phone_link.sh status`** answers in one line: `HTTP 200 (serving)` vs
   `HTTP 530 … origin connection dead`, plus whether the pid is alive.

Two gotchas that cost real debugging time, both handled in the script but worth knowing:

* Probe with **`curl -4`**. `trycloudflare.com` returns AAAA records; a host with broken IPv6
  egress gets connection failures on a *healthy* tunnel, which reads as "tunnel dead" and made
  the supervisor rotate URLs every 25 s. Judge by HTTP code too — a 530 error page is a
  successful fetch of a failure.
* A **brand-new hostname is not immediately reliable.** Measured on this build: right after a
  republish, 8 of 12 sequential requests failed with a connection error while the origin was
  12/12 on localhost; once it had warmed up, 12/12 through the tunnel, twice, on both protocols.
  So the supervisor tolerates extra misses after republishing instead of minting yet another URL,
  and if a freshly printed link refuses to load, wait ~30 s and reload once before concluding
  anything is broken.

**Because a republish changes the URL, never hand out a memorised link mid-demo — send people
to `./phone_link.sh status`, or let them scan the QR** (`web/phone-qr.png` is rewritten on every
republish, and `/phone` renders a QR built from the request's own host, so it is always current).

**Two things to know before you demo with it:**
1. The URL is **ephemeral and unauthenticated** — anyone with the link can read and write the
   demo ward. Fine for judging, wrong to leave running: stop it with `./phone_link.sh stop`
   (avoid `pkill -f cloudflared` in a script — the pattern matches the shell running it and
   kills your own command), and never put real patient data behind it.
2. `trycloudflare.com` is rate-limited and has no uptime guarantee, and some mobile ISPs block
   it outright. Demo-day fallback: run `./run.sh` on a laptop on the same Wi-Fi and open
   `http://<laptop-ip>:8000` — but that is *not* HTTPS, so Chrome blocks the camera unless you
   use `adb reverse tcp:8000 tcp:8000` (localhost counts as secure) or add a self-signed cert.
   The **Simulate fingertip** button needs no camera and no secure context, so it is the safe
   thing to demo first.

---

## Run it (3 commands)

```bash
./run.sh                            # trains if needed, seeds an empty DB, serves everything
```

or explicitly:

```bash
pip install -r requirements.txt
python train.py                     # ✅ steps 6-8: dataset + model + SHAP  (~9 s)
python -m uvicorn vitalguard.server:app --host 0.0.0.0 --port 8000
python -m pytest -q                 # ✅ 25 regression tests
```

Then open **http://localhost:8000/** → *Capture* → **Simulate fingertip**.
That button renders a fingertip clip on a canvas, reads it back through the same
pixel→green-mean path the phone uses, and the server filters it, finds the beats,
scores the risk and explains it.

```bash
python demo.py                      # ✅ whole chain in a terminal, printable report
streamlit run dashboard/app.py      # ✅ step 9: Streamlit + Plotly dashboard (port 8501)
```

To use a real camera, open the same URL on a phone (or laptop) over **HTTPS** —
`getUserMedia` is blocked on plain HTTP. On a LAN, `adb reverse tcp:8000 tcp:8000`
plus `http://localhost:8000` also works for Chrome-on-Android testing.

---

## Status against the 10-step plan

| # | Step | Status | Where |
|---|---|---|---|
| 1 | Camera module + flash | ✅ browser / 🔌 Android code shipped | `web/app.js`, `android/CaptureActivity.kt` |
| 2 | Extract PPG (per-frame intensity) | ✅ | `vitalguard/ppg.py` |
| 3 | Clean signal (SciPy filtering) | ✅ 3rd-order Butterworth band-pass 0.7–3.6 Hz, `filtfilt`, uniform resample, exposure rejection | `ppg.clean_signal` |
| 4 | Heart rate (peaks → BPM) | ✅ three estimators: Welch spectrum, autocorrelation, peak intervals | `heart_rate.analyze` |
| 5 | Signal-quality check + retry | ✅ 0–100 gate; below 60 or <6 dB SNR the reading is **rejected, not scored** | `heart_rate.analyze`, `server.ppg_process` |
| 6 | Patient dataset | ✅ HR + SpO₂/BP/temp + history, correlated cohorts, 18% missing values | `risk_model.make_dataset` |
| 7 | Train AI model | ✅ HistGradientBoostingClassifier, 5-fold CV | `risk_model.train`, `train.py` |
| 8 | Explainability | ✅ exact TreeSHAP (global + per-patient) | `shap.TreeExplainer`, `explain.py` |
| 9 | Dashboard | ✅ both: FastAPI web app **and** Streamlit+Plotly | `web/`, `dashboard/app.py` |
| 10 | Connect everything | ✅ FastAPI + SQLite + alerts | `vitalguard/server.py`, `db.py` |

---

## Measured behaviour (from `python demo.py` on this build)

| check | result |
|---|---|
| Heart-rate accuracy, 120 clean clips, HR 45–190, 12–30 fps, SNR 6–20 dB | **MAE 0.18 BPM**, p90 0.34, **0 readings wrongly trusted** |
| Same at a fixed 30 fps | MAE 0.19 BPM, 116/120 scored, 0 wrongly trusted |
| Extremes, HR 40–200 | MAE 0.17 BPM, 0 wrongly trusted |
| Documented 11-clip set (45/58/63/72/88/97/104/118/133/152/171) | every clip within **1.6 BPM**, 9 of 11 at quality 100 |
| Video file → OpenCV → PPG → HR | 96.0 true → **96.1**, quality 100.0, fold coherence 4.26 |
| 240-frame live-style POST (104 BPM) | 103.9, quality 100.0 |
| Pulseless signal (finger lifted), 30 clips | **30/30 rejected**, max quality 44, retry guidance returned |
| Violent periodic shake (adversarial: artefact 1.6× pulse amplitude) | 69/120 scored, MAE 3.04, p90 2.6; 6 artefact-dominant misreads remain — see limit 3 |
| Clean clips the gate refused anyway | 13/120 at 12–30 fps, 2/120 at 30 fps (low frame rates give too few samples per beat to fold) |
| `Simulate fingertip` button, all 71 rates 60–130 BPM | min quality 66.8, **0 gated**; 12/12 consecutive clicks scored in a real browser |
| Risk model, 5-fold CV on 6000 patients | ROC AUC **0.906**, AP 0.862, Brier 0.104, sensitivity **0.90 @ 90% specificity** |
| Plain NEWS2 baseline on the same target | ROC AUC 0.914 |
| Regression suite | `pytest -q` → **25 passed**, incl. "SHAP sums to the model output" |
| Browser suite | `tests/browser_check.py` → **22 checks passed** on `localhost` *and* through the public tunnel |
| End-to-end scoring latency | < 40 ms (excluding capture) |

### Two algorithm notes worth knowing before you touch `heart_rate.py`

* **Beats are folded by phase, never by a rounded sample count.** `int(round(T*fs))` looks
  harmless, but at 20 fps a 96 BPM heart is 12.45 samples per cycle: truncating to 12 slides
  every successive beat by half a sample, so 19 genuine beats averaged into mush. That single
  line capped clean recordings at quality 58 — below the scoring gate — at 96, 105 and 115 BPM
  and nowhere else, which is exactly the kind of bug that reads as "the app is flaky".
* **`fold_gain` is a veto, not just a bonus.** Measured on 30-clip class samples: real pulses
  fold at 2.8–4.7, a shaking hand at 1.1–1.6, pure noise at 0.2–0.7. A template weaker than
  its own scatter means there is no heartbeat, so `fold_gain < 0.8` caps quality at 44. It also
  has to apply above 10 samples/cycle, or a fabricated 154 BPM on a finger-less lens slips past
  the guard that exists to avoid rewarding degenerate templates.

The design rule the gate enforces: **a number is only produced when it can be
defended.** When two rhythms of similar strength sit in the pulse band — the
signature of a movement artefact — the system refuses and explains why, instead of
reporting whichever peak happened to be taller.

**Read the AUC honestly.** The labels are generated from NEWS2, so ~0.91 is the
*ceiling a rule-derived model can reach*, and the model never sees `news2` as a
feature — that is why it is reported side by side. The claim this prototype
supports is the **pipeline**: optical HR → quality gate → score → explanation →
alert. Point `train.py --dataset ward.csv --target deteriorated_24h` at
MIMIC-IV / eICU / hospital data and the same numbers become the real result.

---

## Architecture

```
┌── phone (Member 1) ──────────┐   ┌── Python server (Member 3) ──────────────┐
│ CameraX + torch              │   │ FastAPI                                  │
│ 12 s @ ~20 fps               │──▶│  /api/ppg/process                        │
│ mean green of centre ROI     │   │   ├ OpenCV (video path) / frames (live)  │
│ 240 × {t, value} ≈ 6 KB      │   │   ├ scipy band-pass 0.7-3.6 Hz         │
└──────────────────────────────┘   │   ├ welch + acf + find_peaks → BPM, HRV │
      or upload .mp4 → /api/ppg/video  │ └ quality gate 0-100 ─────────────┐    │
                                       │                                   ▼    │
                                       │  sklearn HistGB ──▶ risk 0-1      │    │
                                       │  shap.TreeExplainer ─▶ top factors│    │
                                       │  SQLite: patients/readings/alerts ┘    │
                                       └───────────────┬────────────────────────┘
                                                       ▼
                                    /  browser dashboard (FastAPI static)
                                    dashboard/app.py  Streamlit + Plotly
```

Everything the browser and the phone send is the **same JSON**, so swapping the
capture client changes nothing server-side:

```json
{ "frames": [{"t": 0.0, "v": 131.7, "luma": 131.7}, ...],
  "patient_id": 4,
  "vitals": {"spo2": 89, "rr": 29, "sbp": 93, "dbp": 56, "temp_c": 38.9},
  "save": true }
```

## API

| method | path | what |
|---|---|---|
| `POST` | `/api/ppg/process` | frames → HR + quality + risk + SHAP (+ saved reading) |
| `POST` | `/api/ppg/video` | uploaded clip → same, decoded by OpenCV |
| `GET` | `/api/simulate` | synthetic PPG for demos/tests |
| `GET` | `/api/demo/fingertip` | renders a real `.avi` and pushes it through OpenCV |
| `GET` | `/api/patients` · `POST /api/patients` | ward list / upsert |
| `POST` | `/api/patients/{id}/score` | score from entered vitals only |
| `GET` | `/api/patients/{id}/history` | stored readings + explanations |
| `GET` | `/api/alerts` · `POST /api/alerts/{id}/ack` | alert queue |
| `GET` | `/api/metrics` | CV metrics + global SHAP importances |
| `GET` | `/api/snapshot` | one call for the dashboard's 5 s poll |
| `GET` | `/docs` | FastAPI Swagger UI |

## Layout

```
train.py                steps 6-8 CLI (synthetic or --dataset CSV)
demo.py                 end-to-end proof + printable report
vitalguard/ppg.py       steps 1-3   frames → intensity → clean signal
vitalguard/heart_rate.py steps 4-5   BPM, HRV, quality gate, retry guidance
vitalguard/risk_model.py steps 6-8   features, cohorts, training, SHAP
vitalguard/explain.py    step 8      SHAP → sentences → escalation advice
vitalguard/db.py         step 9      SQLite: patients, readings, alerts
vitalguard/server.py     step 10     FastAPI wiring
web/                     steps 1,9   capture page + dashboard (vanilla JS, no build step)
dashboard/app.py         step 9      Streamlit + Plotly
android/                 step 1      CameraX + Kotlin reference 🔌
```

---

## Team split, who touches what

* **Member 1 — camera/PPG/HR:** `vitalguard/ppg.py`, `vitalguard/heart_rate.py`, `android/`.
  Task: record 12 s on a real finger, compare the returned BPM against a pulse oximeter,
  tune `BANDPASS_HZ` and the ROI size. Add `ImageCapture` high-speed mode if the phone supports it.
* **Member 2 — AI/risk/SHAP:** `vitalguard/risk_model.py`, `vitalguard/explain.py`, `train.py`.
  Task: replace synthetic labels with MIMIC-IV or eICU, keep `FEATURES` as the contract,
  recalibrate the alert tiers in `explain.TIERS`.
* **Member 3 — backend/DB/dashboard:** `vitalguard/server.py`, `db.py`, `web/`, `dashboard/`.
  Task: auth + TLS, ward filtering, recheck timers, and the `POST /api/patients/{id}/score`
  retry loop the nurse flow needs.

## If it doesn't work

Symptoms we have actually hit and fixed — check these first:

| symptom | cause | fix |
|---|---|---|
| every button does nothing / red `422` | old `app.js` held in cache, or a POST sent without a JSON content-type | reload once (now impossible: `Cache-Control: no-cache` is set on `/`, `/static/*`, `/phone`) |
| "no camera" on the phone | page opened over plain `http://` | `isSecureContext` must be true — use the `https://` tunnel link, or `adb reverse tcp:8000 tcp:8000` and open `http://localhost:8000` |
| camera button works but *Record* captures 0 frames | the `<video>` had no frame yet, so the ROI was 0×0 and `getImageData` threw, killing the loop | fixed: `roiBox()` returns "not ready" until `videoWidth > 0`, and the sampler is wrapped in try/catch so one bad frame can't end the capture |
| link from a WhatsApp/Instagram message won't use the camera | in-app browsers strip `getUserMedia` | open in Chrome/Safari from the browser menu |
| "Not scored: quality …" after recording | that is the gate working — finger slipping, flash over-exposing, or motion | press firmly over the lens, back the flash off, rest the hand on a table |

Two automated ways to prove the client is healthy, both runnable by anyone on the team:

```bash
python -m pytest -q                      # 25 pipeline tests
pip install playwright && python -m playwright install chromium
python tests/browser_check.py https://<your-tunnel-url>   # 22 browser checks
```

`browser_check.py` loads the page in a real headless Chromium at a **phone viewport**, fails
on any uncaught JS error, clicks every tab, checks the waveform and gauge canvases actually
have pixels, and drives the **real `getUserMedia` path** against a fake camera device — so it
exercises the exact code a phone runs, including the frame-sampling loop.

## Prototype limits — say them before a judge does

1. **Optical SpO₂ is not implemented.** Real pulse oximetry needs red *and* infrared and
   the ratio-of-ratios; a single white-LED camera gives HR, not SpO₂. SpO₂/RR/BP/temp are
   entered or read from a probe. RR is *estimated* from the PPG envelope and labelled as such.
2. **Labels are synthetic** (NEWS2-derived) — the model proves the pipeline, not clinical accuracy.
3. **Motion is the hard limit, and we measured the boundary.** Refusal rate vs artefact
   amplitude (30 clips per row, artefact placed off the pulse harmonics, HR 70–100):
   artefact **0.7–1.0× the pulse → 30/30 refused**, 1.3× → 25/30 refused, and at
   **1.6×+ the spectrum collapses to a single peak — 28/30 get scored and are wrong**
   (MAE 75). That top regime is not a threshold problem: with one channel and no second peak
   to compare against, the artefact *is* the dominant periodic component. A second-harmonic
   test (a real pulse has one, a mechanical shake is near-sinusoidal) does separate the two on
   synthetic data, but its distribution overlaps legitimately sinusoidal pulses, so using it as
   a gate would refuse real fingers to fix an artefact of our own simulator — we measured it
   (clean 0.035–0.214 vs weak-harmonic pulses 0.002–0.033) and deliberately did not ship it.
   Wearables resolve this with an accelerometer for adaptive noise cancellation. Not built in
   time: accelerometer reference, two consecutive agreeing reads, dual wavelength. The
   **shaky finger** checkbox on the Capture tab drives exactly this test — it is pinned to
   96 BPM against a 64 BPM shake so the refusal is deterministic (6/6 in-browser), and the
   boundary above is what "honest about its limits" means for this device.
4. **Android files are unbuilt here** — no Android SDK in this environment. They are complete,
   consistent reference code (CameraX, torch, YUV green plane, Retrofit) that needs an actual
   build in Android Studio.
5. **Not a medical device.** No FDA/CE path, no emergency override policy; the UI states that
   clinician judgement supersedes the model, and the model refuses to score low-quality data.

## Roadmap to something real

calibrate optical HR against a reference oximeter on ≥20 subjects → train on MIMIC-IV/eICU with
patient-level splits → add red+IR or a BLE oximeter for true SpO₂ → temporal model (GRU/Transformer
over the last 6 h of readings) instead of single-reading scoring → SHAP stability tests → HL7/FHIR
export → FDA Class II pathway if it ever leaves the ward as advice.
