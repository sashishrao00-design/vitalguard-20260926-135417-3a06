"""
Steps 1-3 of the VitalGuard AI build plan.

1. Camera module  -> in the browser prototype frames come from getUserMedia; on the phone
                      from CameraX (see android/CaptureActivity.kt); in the offline demo
                      from an .avi/.mp4 the server decodes with OpenCV.
2. Extract PPG    -> mean pixel intensity of the region of interest per frame.
3. Clean signal   -> saturation/exposure rejection, detrending, Butterworth bandpass.

This module only deals with *photoplethysmography* physics: light -> intensity series.
Heart-rate math lives in heart_rate.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps

# ---------------------------------------------------------------------------
# Constants (documented so the team can tune them during the demo)
# ---------------------------------------------------------------------------
BANDPASS_HZ = (0.7, 3.6)   # 42 - 216 BPM; kills breathing (<0.5 Hz) and HF noise
SATURATION_LUMA = 246.0    # 8-bit luma above which a frame is clipped / over-exposed
MIN_PERFUSION = 0.0025     # AC/DC ratio below which there is essentially no pulse


@dataclass
class PPGSeries:
    """A cleaned, uniformly resampled PPG signal ready for heart-rate analysis."""

    t: np.ndarray               # seconds, uniform grid
    y: np.ndarray               # band-passed, unit-normalised AC component
    fs: float                   # effective sampling rate of the uniform grid
    raw_mean: float             # mean luma of the ROI (0-255)
    ac_dc_ratio: float          # perfusion index proxy
    n_frames_in: int            # frames offered
    n_frames_used: int          # frames surviving exposure rejection
    rejected: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


# ---------------------------------------------------------------------------
# Frame -> intensity (step 2)
# ---------------------------------------------------------------------------
def roi_intensity_from_frame(frame: np.ndarray, box: tuple[int, int, int, int] | None = None) -> float:
    """Mean *green* channel value of the ROI.

    Green (~540 nm) gives the strongest pulse-to-noise ratio for fingertip PPG
    under white/LED illumination, so we use it by default and fall back to luma.
    ``frame`` is an HxWx3 uint8 BGR array (OpenCV convention).
    """
    if frame.ndim == 3:
        g = frame[..., 1].astype(np.float64)
    else:
        g = frame.astype(np.float64)
    if box is not None:
        x, y, w, h = (int(v) for v in box)
        if w > 2 and h > 2:
            g = g[y : y + h, x : x + w]
    return float(g.mean())


def series_from_video(path: str | Path, fps_limit: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Decode a fingertip clip and return (timestamps, mean ROI intensity).

    Used by the offline demo and by the Android upload path, where the phone
    hands us a recorded video instead of per-frame numbers.
    """
    import cv2  # imported lazily so the ML code still runs without opencv

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps_limit)))
    t: list[float] = []
    v: list[float] = []
    i = 0
    # centre-crop 45% of the frame: on the phone the finger fills the middle
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            box = (int(w * 0.275), int(h * 0.275), int(w * 0.45), int(h * 0.45))
            t.append(i / src_fps)
            v.append(roi_intensity_from_frame(frame, box))
        i += 1
    cap.release()
    if len(v) < 8:
        raise ValueError("video too short to contain a pulse signal")
    return np.asarray(t, dtype=np.float64), np.asarray(v, dtype=np.float64)


# ---------------------------------------------------------------------------
# Cleaning (step 3)
# ---------------------------------------------------------------------------
def _reject_bad_frames(t: np.ndarray, luma: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop over-exposed and pitch-black frames: the flash blowing out the skin or
    the finger slipping off the lens produces huge non-physiological jumps."""
    clipped = luma > SATURATION_LUMA
    dark = luma < 12.0
    keep = ~(clipped | dark)
    info = {
        "n_clipped": int(clipped.sum()),
        "n_dark": int(dark.sum()),
        "frac_clipped": float(clipped.mean()),
    }
    if keep.sum() < max(16, 0.4 * len(keep)):
        # too much rejection -> fall back to the median-clamped series so the
        # caller still gets a number and a poor quality score instead of a crash
        info["fallback"] = "clamped_instead_of_dropped"
        capped = np.clip(luma, None, SATURATION_LUMA)
        return t, capped, info
    return t[keep], luma[keep], info


def clean_signal(
    t_raw: np.ndarray,
    v_raw: np.ndarray,
    bandpass: tuple[float, float] = BANDPASS_HZ,
) -> PPGSeries:
    """Raw (timestamp, intensity) pairs -> clean, uniform, band-passed PPG."""
    t_raw = np.asarray(t_raw, dtype=np.float64).ravel()
    v_raw = np.asarray(v_raw, dtype=np.float64).ravel()
    n_in = len(t_raw)
    if n_in < 16:
        raise ValueError(f"need >=16 frames, got {n_in}")

    order = np.argsort(t_raw)
    t_raw, v_raw = t_raw[order], v_raw[order]
    dt = np.diff(t_raw)
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("all timestamps are identical")
    fs_nom = float(1.0 / np.median(dt))

    t, v, rejected = _reject_bad_frames(t_raw, v_raw)
    if len(t) < 16:
        raise ValueError("not enough usable frames after exposure rejection")

    dc = float(np.mean(v))
    # resample onto a uniform grid: camera callbacks jitter badly on phones
    fs = float(min(max(fs_nom, 5.0), 30.0))
    t_u = np.arange(t[0], t[-1], 1.0 / fs)
    if len(t_u) < 16:
        raise ValueError("signal too short after resampling")
    v_u = np.interp(t_u, t, v)

    # detrend (slow vasomotor drift) with a robust high-pass, then band-pass
    nyq = fs / 2.0
    lo = bandpass[0] / nyq
    hi = min(bandpass[1] / nyq, 0.95)
    lo = max(lo, 0.005)
    b, a = sps.butter(3, [lo, hi], btype="band")
    y = sps.filtfilt(b, a, v_u, padlen=min(3 * max(len(b), len(a)), len(v_u) - 1))

    ac = float(np.std(y))
    rejected["raw_std"] = float(np.std(v_u))
    rejected["n_frames_used"] = int(len(v_u))

    return PPGSeries(
        t=t_u,
        y=y / (ac + 1e-12),
        fs=fs,
        raw_mean=dc,
        ac_dc_ratio=ac / (dc + 1e-12),
        n_frames_in=n_in,
        n_frames_used=int(len(t)),
        rejected=rejected,
    )


def synthetic_recording(
    hr_bpm: float = 74.0,
    seconds: float = 12.0,
    fps: float = 20.0,
    snr_db: float = 12.0,
    motion_bpm: float = 0.0,
    seed: int | None = 1,
    jitter_ms: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a realistic fingertip intensity series.

    Two purposes: (a) unit-test the extractor with a known ground-truth HR,
    (b) power the "simulate fingertip" button so the prototype demos on a laptop
    with no camera permission and on a judge's phone with no finger attached.
    A cardiac pulse is modelled as a fundamental plus harmonics, on top of a
    baseline with breathing drift, motion artefact and shot noise.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps + rng.normal(0, jitter_ms / 1000.0, n)
    f = hr_bpm / 60.0
    phase = 2 * np.pi * f * t
    pulse = (
        np.sin(phase)
        - 0.42 * np.sin(2 * phase + 0.5)
        + 0.20 * np.sin(3 * phase + 1.1)
        + 0.08 * np.sin(4 * phase)
    )
    pulse = pulse / np.std(pulse)

    baseline = 150.0 + 6.0 * np.sin(2 * np.pi * 0.25 * t)          # breathing / drift
    amplitude = baseline * 0.012                                    # ~1.2% AC, typical finger
    if motion_bpm > 0:
        pulse += 1.6 * np.sin(2 * np.pi * (motion_bpm / 60.0) * t) + rng.normal(0, 0.8, n)
    noise = rng.normal(0, 1.0, n) * amplitude / (10 ** (snr_db / 20.0))
    shot = rng.normal(0, 0.5, n) * np.sqrt(np.maximum(baseline, 1.0)) * 0.02
    v = baseline + amplitude * pulse + noise + shot
    return t, np.clip(v, 0, 255)


def write_demo_video(path: str | Path, **kwargs) -> dict:
    """Render a fingertip clip (pulsing brightness + noise) as .avi for OpenCV.

    Lets us prove step 1-2 end to end with zero camera permissions: the server
    reads real JPEG/PNG frames out of a real container.
    """
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(kwargs.get("seed", 7))
    hr = kwargs.get("hr_bpm", 74.0)
    fps = kwargs.get("fps", 20)
    secs = kwargs.get("seconds", 12.0)
    n = int(secs * fps)
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    finger = np.clip(1.0 - (r / (0.42 * w)) ** 2, 0, 1)
    t = np.arange(n) / fps
    f = hr / 60.0
    phase = 2 * np.pi * f * t
    pulse = np.sin(phase) - 0.4 * np.sin(2 * phase + 0.5) + 0.18 * np.sin(3 * phase)
    pulse = pulse / np.std(pulse)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(p), fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        p = p.with_suffix(".avi") if p.suffix != ".avi" else p
        out = cv2.VideoWriter(str(p), fourcc, fps, (w, h))
    for i in range(n):
        base = 118.0 + 4.4 * pulse[i] + rng.normal(0, 1.6)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for c, tint in enumerate((150, 92, 96)):  # BGR: reddish skin
            img[..., c] = np.clip(finger * (base * (1 + 0.06 * (c == 0)) * (tint / 110.0)), 0, 255)
        img = img + rng.normal(0, 2.0, (h, w, 1)) * finger[..., None]
        out.write(np.clip(img, 0, 255).astype(np.uint8))
    out.release()
    return {"path": str(p), "frames": n, "fps": fps, "seconds": secs, "true_hr_bpm": hr}
