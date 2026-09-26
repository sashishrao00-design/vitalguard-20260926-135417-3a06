# Signal-safety-2 update

## What changed
- Normal demo generates a reproducible **80 BPM** synthetic input. The estimated
  BPM still comes from the unchanged SciPy/heart-rate algorithm; it is never
  replaced with the input value.
- A simulation estimate more than 3 BPM away from its known input is refused
  before risk scoring. This truth check does not apply to physical camera data.
- Demo output is explicitly labelled **not your heart rate**; synthetic capture
  requests cannot save readings to patients, even if saving is requested.
- Rejected estimates are diagnostics, not displayed measurements. The main HR,
  HRV and breathing values are cleared; no risk or SHAP factors are shown.
- Demo and camera requests are mutually exclusive, stale response rendering is
  guarded, and the previous result is hidden while a new one is processing.
- Starting a camera after a demo restores the hidden video and recomputes its ROI.
  Starting a demo stops the camera stream.
- Removed silent simulation re-rolls. A refused signal is reported as refused.
- Technical estimator candidates are collapsed and explicitly not measurements.
- `/api/health` reports `app_version: signal-safety-2`.

## Evidence and limits
The reported 196 BPM demo result was not reproduced on the deployed old build:
40 ordinary simulations produced 60.0–129.7 BPM, with maximum error 0.5 BPM.
This update addresses independently verified display/lifecycle/integrity problems;
it does not establish the root cause of that specific report.

Checks on the updated local build:
- 32 pytest checks passed, including backend rejection of an injected 196 BPM
  estimate for an 80 BPM synthetic input.
- Existing browser suite and end-to-end API tests passed.
- Browser fault injection of a 196 BPM answer marked high-quality was refused.
- Repeated demos: actual estimated 79.7 BPM for an 80 BPM test signal.
- Concurrent requests, refusal display, synthetic saving, and camera/demo
  transitions verified with an automated browser and a fake camera stream.

This is not clinical validation, nor a physical-phone camera test. Coherent motion
can still mimic a pulse in a single optical channel. Genuine high heart rates are
not clamped to a normal range. Do not use this prototype for medical decisions.

## Deploy to the existing service
Use the supplied `VitalGuard-Update.ipynb` in Colab to upload this patch to your
existing GitHub repository. It updates only the named files, validates their
previous versions, and will stop rather than overwrite conflicting changes.
Then in the existing Render service choose Manual Deploy → Deploy latest commit.
No new hosting service is required.

After Render says Live, open `/api/health` and confirm
`"app_version":"signal-safety-2"`, then refresh the main page. The button should
say **Run demo · 80 BPM**. A public health check proves deployment version only;
it does not establish clinical accuracy.
