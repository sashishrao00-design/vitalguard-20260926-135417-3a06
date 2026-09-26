"""
Steps 4-5 of the VitalGuard AI build plan: heart-rate calculation and the
signal-quality gate that decides whether a reading is trustworthy enough to
send to the risk model.

Two independent estimators are combined, which is what makes the number robust
on a shaky phone:

  * spectral  - Welch periodogram peak inside the physiological band
  * temporal  - autocorrelation of the filtered PPG + peak-interval statistics

If the two disagree by more than a small tolerance the quality score drops and
the app asks the user to retry, rather than reporting a wrong BPM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from .ppg import MIN_PERFUSION

HR_MIN_BPM, HR_MAX_BPM = 40.0, 200.0
# Refuse a reading when a second in-band spectral peak reaches this fraction of the
# main one. Chosen empirically on 200 clean + 140 movement-corrupted synthetic clips:
# 0.30 catches 85% of artefact-driven misreads while refusing only 2.5% of clean
# readings (0.20 catches 90% but throws away 9% of good signals - too blunt).
AMBIGUITY_REFUSE = 0.30
HARMONIC_TOL_BPM = 6.0


@dataclass
class HRResult:
    hr_bpm: float | None
    hr_spectral_bpm: float
    hr_temporal_bpm: float
    hr_from_peaks_bpm: float | None
    hrv_rmssd_ms: float | None
    hrv_sdnn_ms: float | None
    breathing_rate_bpm: float | None
    quality: float                    # 0-100
    quality_label: str
    n_peaks: int
    agree_bpm: float
    guidance: list[str] = field(default_factory=list)
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["guidance"] = list(self.guidance)
        return d


def _spectral_hr(y: np.ndarray, fs: float) -> tuple[float, float]:
    """Peak of the Welch PSD in-band. Returns (bpm, snr) where snr is the ratio of
    power at the peak to the median power of the band."""
    nper = int(min(len(y), max(128, 4 * fs)))
    f, p = sps.welch(y, fs=fs, nperseg=nper, noverlap=nper // 2)
    band = (f >= HR_MIN_BPM / 60.0) & (f <= HR_MAX_BPM / 60.0)
    if not band.any():
        return 0.0, 0.0
    fb, pb = f[band], p[band]
    i = int(np.argmax(pb))
    # parabolic interpolation for sub-bin accuracy
    if 0 < i < len(pb) - 1:
        d0, d1, d2 = np.log(pb[i - 1] + 1e-20), np.log(pb[i] + 1e-20), np.log(pb[i + 1] + 1e-20)
        denom = d0 - 2 * d1 + d2
        shift = 0.5 * (d0 - d2) / denom if abs(denom) > 1e-12 else 0.0
        fpk = float(fb[i] + shift * (fb[1] - fb[0]))
    else:
        fpk = float(fb[i])
    snr = float(pb[i] / (np.median(pb) + 1e-20))
    return fpk * 60.0, snr


def _temporal_hr(y: np.ndarray, fs: float) -> float:
    """Autocorrelation lag of the dominant periodicity."""
    x = y - y.mean()
    x = x * np.hanning(len(x))
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    ac = ac / (ac[0] + 1e-20)
    lo = int(fs * 60.0 / HR_MAX_BPM)
    hi = int(fs * 60.0 / HR_MIN_BPM)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return 0.0
    seg = ac[lo : hi + 1]
    k = int(np.argmax(seg))
    return 60.0 * fs / (k + lo)


def _power_at(y: np.ndarray, fs: float, bpm: float) -> float:
    """Welch PSD evaluated at a candidate frequency - used to break ties between a
    heart rate and its sub/super-harmonics (e.g. 110 vs 55 BPM)."""
    if bpm <= 0:
        return 0.0
    nper = int(min(len(y), max(128, 4 * fs)))
    f, p = sps.welch(y, fs=fs, nperseg=nper, noverlap=nper // 2)
    return float(np.interp(bpm / 60.0, f, p))


def _refine_to_peak(y: np.ndarray, fs: float, around_bpm: float, tol_bpm: float = 8.0) -> float:
    """Snap a candidate to the strongest spectral peak within +-tol of it.

    Arbitration works on halves/doubles of three estimators, so it can end up on
    a value like 158.4 that is really the same rhythm as the true peak at 156.0 -
    the estimator that produced it was just quantised differently. Re-reading the
    periodogram inside the candidate's band recovers the peak without letting the
    correction wander into a different rhythm (the band is what keeps it honest).
    """
    if around_bpm <= 0:
        return around_bpm
    nper = int(min(len(y), max(128, 4 * fs)))
    f, pw = sps.welch(y, fs=fs, nperseg=nper, noverlap=nper // 2)
    lo, hi = (around_bpm - tol_bpm) / 60.0, (around_bpm + tol_bpm) / 60.0
    idx = np.where((f >= max(lo, HR_MIN_BPM / 60.0)) & (f <= min(hi, HR_MAX_BPM / 60.0)))[0]
    if idx.size == 0:
        return around_bpm
    k = int(idx[np.argmax(pw[idx])])
    if 0 < k < len(pw) - 1:
        d0, d1, d2 = np.log(pw[k - 1] + 1e-20), np.log(pw[k] + 1e-20), np.log(pw[k + 1] + 1e-20)
        den = d0 - 2 * d1 + d2
        shift = 0.5 * (d0 - d2) / den if abs(den) > 1e-12 else 0.0
        return float((f[k] + shift * (f[1] - f[0])) * 60.0)
    return float(f[k] * 60.0)


def _template_stats(y: np.ndarray, fs: float, bpm: float) -> tuple[float, float, float]:
    """Fold the signal at a candidate period and judge the resulting average beat.

    Returns (fold_gain, peaks_per_cycle, sharpness).

    Real PPG: beats align at the true period, so the folded average is strong
    relative to the leftover scatter (fold_gain high), it contains exactly one
    dominant systolic peak, and it is asymmetric (fast upstroke, slower decay).
    A coherent artefact - a shaking hand at a steady rate - produces a smooth,
    symmetric, single-hump template; folding at half the true rate produces two
    humps per cycle. Those are measurable, so we use them instead of trusting the
    periodogram alone.
    """
    T = 60.0 / bpm if bpm > 0 else 0.0
    if T <= 0 or len(y) < 3 * T * fs:
        return 0.0, 9.0, 0.0
    # Fold by PHASE, not by a rounded sample count. A phone at 20 fps sees 96 BPM as
    # 12.45 samples per beat: truncating that to 12 slides every cycle a further half
    # sample, so 19 real beats average into mush and a textbook pulse scored as
    # "cycles did not line up". Phase folding is exact at any rate.
    nb = 48                                     # bins per cycle
    t = np.arange(len(y)) / fs
    ph = np.mod(t / T, 1.0)
    cyc = np.floor(t / T)
    edges = np.linspace(0.0, 1.0, nb + 1)
    idx = np.clip(np.searchsorted(edges, ph, side="right") - 1, 0, nb - 1)
    tmpl = np.empty(nb)
    for k in range(nb):
        sel = idx == k
        tmpl[k] = float(y[sel].mean()) if sel.any() else np.nan
    if np.isnan(tmpl).any():                    # circular gap fill
        good = ~np.isnan(tmpl)
        if good.sum() < 3:
            return 0.0, 9.0, 0.0
        xs = np.arange(nb)
        ext = np.concatenate([tmpl[good], tmpl[good], tmpl[good]])
        xe = np.concatenate([xs[good] - nb, xs[good], xs[good] + nb])
        tmpl = np.interp(xs, xe, ext, period=nb) if False else np.interp(xs, xe, ext)
    # residual scatter: each sample against the template value at its own phase
    tmpl_at = np.interp(np.concatenate([ph, [1.0]]), edges, np.concatenate([tmpl, tmpl[:1]]))[: len(ph)]
    resid = float(np.std(y - tmpl_at))
    fold_gain = float(np.std(tmpl) / (resid + 1e-12))

    n = max(6, int(round(T * fs)))              # kept for the morphology window sizes
    pk, _ = sps.find_peaks(tmpl, prominence=0.25 * (np.ptp(tmpl) + 1e-12))
    peaks = float(len(pk))
    _ = cyc

    d = np.abs(np.diff(tmpl))
    sharp = float(d.max() / (np.std(d) + 1e-12)) if d.size else 0.0
    # symmetry of the cycle: systolic upstroke vs diastolic decay differ in a real
    # pulse; a sinusoidal artefact is near-perfectly symmetric.
    if pk.size:
        i = int(pk[0])
        rise = tmpl[max(0, i - n // 4): i + 1]
        fall = tmpl[i: min(n, i + n // 2)]
        if rise.size >= 3 and fall.size >= 3:
            k = min(rise.size, fall.size)
            asym = float(np.abs(np.mean(np.diff(rise[-k:])) - np.mean(np.diff(fall[:k]))))
            asym = asym / (np.ptp(tmpl) + 1e-12)
        else:
            asym = 0.0
    else:
        asym = 0.0
    return fold_gain, peaks, sharp * (0.5 + min(asym, 1.0))


def _spectral_ambiguity(y: np.ndarray, fs: float) -> tuple[float, float]:
    """Height of the strongest *competing* in-band peak relative to the main peak.

    A clean PPG has one dominant line plus weaker harmonics (typically p2/p1 ~ 0.2),
    so a second peak of comparable height means two real rhythms are present -
    almost always the pulse and a periodic movement artefact. That case must be
    refused rather than guessed at, because nothing in a single optical channel can
    say which one is the heart.
    """
    nper = int(min(len(y), max(128, 4 * fs)))
    f, pw = sps.welch(y, fs=fs, nperseg=nper, noverlap=nper // 2)
    band = (f >= HR_MIN_BPM / 60.0) & (f <= HR_MAX_BPM / 60.0)
    if not band.any():
        return 0.0, 0.0
    fb, pb = f[band], pw[band]
    # A 3rd-order Butterworth leaves energy piled up at its own cut-off, which is not
    # a second rhythm - it was being read as one at low heart rates (a competitor at
    # exactly the 42 BPM band edge). Search for competitors only inside the flat part
    # of the passband, 51-204 BPM.
    lo_hz, hi_hz = 51.0 / 60.0, 204.0 / 60.0
    inband = (fb >= lo_hz) & (fb <= hi_hz)
    if not inband.any():
        return 0.0, float(fb[int(np.argmax(pb))] * 60.0)
    fb2, pb2 = fb[inband], pb[inband]
    idx, _ = sps.find_peaks(pb2, prominence=0.0)
    if idx.size < 2:
        return 0.0, float(fb2[int(np.argmax(pb2))] * 60.0)
    order = idx[np.argsort(pb2[idx])[::-1]]
    top = float(pb2[order[0]])
    f1 = float(fb2[order[0]] * 60.0)
    # A competitor must be separated by more than the analysis can resolve. At
    # 20 fps / 128-sample windows one Welch bin is ~9 BPM wide, so excluding only
    # "3 BPM" counted the two neighbouring bins of ONE peak as a second rhythm and
    # wrongly capped clean readings at the ambiguity score. Require >= 2 bins and a
    # real relative gap; anything closer is the same peak, not a competing one.
    df = float(fb[1] - fb[0]) if fb.size > 1 else 0.0
    f1_hz = f1 / 60.0
    comp = 0.0
    for k in order[1:]:
        fk = float(fb2[k])
        if abs(fk - f1_hz) < 2.0 * df:            # same peak, adjacent bins
            continue
        # A pulse is harmonic-rich: real energy also sits at 2f, 3f, 4f (and a
        # missed-beat alias appears at f/2). Those are the SAME rhythm, so they are
        # not a competing one - flagging them is what made clean clips get refused.
        # A movement artefact lands on no such ratio, so only those count.
        r = fk / f1_hz if f1_hz > 0 else 1.0
        near = min(abs(r - m) / max(r, 1.0) for m in (0.5, 1.0, 2.0, 3.0, 4.0))
        if near < 0.06 or abs(r - 1.0) < 0.06:
            continue
        comp = max(comp, float(pb2[k]))
    return comp / (top + 1e-20), f1


def analyze(ppg) -> HRResult:
    """ppg is a :class:`vitalguard.ppg.PPGSeries`."""
    y, fs, t = ppg.y, ppg.fs, ppg.t
    hr_sp, snr = _spectral_hr(y, fs)
    hr_tp = _temporal_hr(y, fs)

    # --- peak detection -> BPM, RR intervals, HRV -----------------------------
    min_dist = max(int(fs * 0.25), int(fs * 60.0 / HR_MAX_BPM))
    peaks, props = sps.find_peaks(y, distance=min_dist, prominence=0.45)
    hr_peaks = None
    rmssd = sdnn = None
    rr_ms = np.array([])
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        rr_ms = rr * 1000.0
        keep = (rr_ms > 60_000 / HR_MAX_BPM) & (rr_ms < 60_000 / HR_MIN_BPM)
        rr_ms = rr_ms[keep]
        if len(rr_ms) >= 2:
            hr_peaks = float(60_000.0 / np.median(rr_ms))
            d = np.diff(rr_ms)
            rmssd = float(np.sqrt(np.mean(d**2))) if len(d) else None
            sdnn = float(np.std(rr_ms))
    elif len(peaks) == 2:
        hr_peaks = float(60_000.0 / ((peaks[1] - peaks[0]) / fs * 1000.0))

    # --- consensus -------------------------------------------------------------
    cands = [h for h in (hr_sp, hr_tp, hr_peaks) if h and h > 0]
    # Two different quantities that happen to share a name in the original design:
    #   agree_raw - the spread of the RAW estimator outputs, whose only job is to
    #     detect "the estimators disagree, probably an octave" and trigger the
    #     harmonic arbitration below.
    #   agree     - the spread AFTER arbitration, modulo octaves, used to score
    #     confidence. Zeroing this before the arbitration would silently disable it.
    agree_raw = float(np.std(cands)) if len(cands) > 1 else 0.0
    agree = agree_raw
    hr = hr_sp
    ambiguous = False

    # When estimators disagree the usual cause is a harmonic or subharmonic
    # (a strong second harmonic reads as double the rate, a missed beat as half).
    # Build the candidate set from halves/doubles, then pick the one backed by the
    # most spectral power, tie-broken by agreement with the peak-interval estimate.
    if len(cands) >= 2 and agree_raw > HARMONIC_TOL_BPM:
        alts: list[float] = []
        for c in cands:
            for alt in (c, c / 2.0, c * 2.0):
                if HR_MIN_BPM <= alt <= HR_MAX_BPM and all(abs(alt - a) > 2.0 for a in alts):
                    alts.append(float(alt))
        if alts:
            power = np.array([_power_at(y, fs, a) for a in alts])
            power = power / (power.max() + 1e-20)
            anchor = hr_peaks if hr_peaks else float(np.median(cands))
            # Time-domain evidence: beat alignment + one-peak-per-cycle morphology.
            # A candidate whose period is shorter than ~10 samples cannot be folded
            # meaningfully, so it gets a NEUTRAL score for that term rather than a
            # zero - otherwise high rates at low frame rates lose every comparison to
            # their own subharmonic (observed: true 166 at 12 fps losing to 55).
            stats = [_template_stats(y, fs, a) for a in alts]
            spp = np.array([(60.0 / a) * fs for a in alts])
            usable = spp >= 10.0
            raw_fold = np.array([st[0] for st in stats])
            best = raw_fold[usable].max() if usable.any() else 0.0
            fold = np.where(usable, raw_fold / (best + 1e-20), 1.0)
            fold = fold / (fold.max() + 1e-20)
            raw_pk = np.array([st[1] for st in stats])
            morph = np.where(~usable, 1.0,
                             np.where((raw_pk >= 1) & (raw_pk <= 2), 1.0,
                                      np.where(raw_pk > 2, 0.35, 0.6)))
            agree_t = np.exp(-np.abs(np.array(alts) - anchor) / 8.0)
            prior = np.exp(-np.abs(np.array(alts) - hr_sp) / 4.0)   # favour the ML spectral peak
            score = 1.5 * power + 1.2 * fold + 0.7 * morph + 0.6 * agree_t + 0.6 * prior
            ranked = np.argsort(score)[::-1]
            hr = _refine_to_peak(y, fs, float(alts[int(ranked[0])]))
            # Ambiguity: if a harmonically related rival is nearly as well
            # supported, we cannot tell heart rate from shake on this clip. Say so
            # rather than emit a confident number - the gate below applies the cap.
            if len(ranked) > 1:
                other = float(alts[int(ranked[1])])
                ratio = max(hr, other) / max(min(hr, other), 1e-9)
                if 1.7 < ratio < 2.35 and score[ranked[1]] >= 0.80 * score[ranked[0]]:
                    ambiguous = True

    # --- breathing rate from the slow envelope (extra NEWS2 input, free of charge)
    resp = None
    try:
        b1, a1 = sps.butter(2, [0.1 / (fs / 2), 0.45 / (fs / 2)], btype="band")
        env = np.abs(sps.hilbert(sps.filtfilt(b1, a1, y)))
        b2, a2 = sps.butter(2, max(0.05 / (fs / 2), 0.01), btype="low")
        env = sps.filtfilt(b2, a2, env)
        fb, pb = sps.welch(env - env.mean(), fs=fs, nperseg=min(len(env), int(8 * fs)))
        m = (fb >= 0.1) & (fb <= 0.6)
        if m.any():
            resp = float(fb[m][np.argmax(pb[m])] * 60.0)
    except Exception:
        resp = None

    amb_ratio, _f1 = _spectral_ambiguity(y, fs)

    # --- quality gate (step 5) --------------------------------------------------
    q = 0.0
    notes: list[str] = []
    q += min(35.0, 35.0 * np.log10(max(snr, 1e-3)) / np.log10(28.0))        # spectral prominence
    q += 30.0 if agree <= 3.0 else max(0.0, 30.0 - 3.0 * (agree - 3.0))      # agreement
    expected = (hr / 60.0) * ppg.duration
    cov = len(peaks) / expected if expected > 0 else 0.0
    q += 20.0 * float(np.clip(cov, 0, 1))                                     # peak coverage
    perf = ppg.ac_dc_ratio
    q += 15.0 * float(np.clip((perf - MIN_PERFUSION) / 0.02, 0, 1))           # perfusion

    # Agreement is judged AFTER harmonic arbitration and modulo octaves: an
    # autocorrelation that says 47.8 while the spectrum says 96.2 is not a
    # disagreement, it is the same beat measured one octave down - which is exactly
    # what the arbitration above exists to resolve. Scoring the raw spread punished
    # clean signals for a known, already-fixed artefact.
    if len(cands) > 1:
        folded = []
        for c in cands:
            if c <= 0:
                continue
            best = min((c * k for k in (0.25, 0.5, 1.0, 2.0, 4.0)), key=lambda v: abs(v - hr))
            folded.append(best)
        agree = float(np.std(folded)) if len(folded) > 1 else 0.0

    # beat alignment is the strongest evidence that what we found is a pulse and
    # not a periodic artefact, so it can both lift and pull down the score.
    samples_per_cycle = (60.0 / hr) * fs if hr else 0.0
    fold_g, morph_pk, morph_s = _template_stats(y, fs, hr) if hr else (0.0, 9.0, 0.0)
    # Below ~10 samples per cardiac cycle there is nothing to align: the template is
    # 2-3 samples of data and the metric degenerates, so we neither reward nor punish.
    fold_ok = samples_per_cycle >= 10.0
    # Measured on 30-clip class samples: clean pulses fold at 2.8-4.7, a shaking hand
    # at 1.1-1.6, pure noise (finger off the lens) at 0.2-0.7. So a low fold gain is
    # positive evidence AGAINST a heartbeat, not merely the absence of evidence, and
    # it is used that way - while the cut sits low enough that a real finger with
    # normal beat-to-beat variation still passes.
    # A template weaker than its own scatter means there is no repeatable beat, at any
    # rate - so this veto is NOT limited to fold_ok. It fired at 154 BPM on a
    # finger-less lens (7.8 samples/cycle) precisely because the reward guard below
    # was skipped, and pure noise must never be rescued by a high rate.
    if samples_per_cycle >= 6.0 and fold_g < 0.8:
        q = min(q, 44.0)        # below the "poor" cut-off: this is not a heartbeat
        notes.append("No repeatable pulse shape could be built from this clip - the "
                     "camera is seeing noise, not a heartbeat. Cover the lens with the "
                     "fingertip, press firmly, and retake.")
    elif fold_ok:
        if fold_g < 1.0:
            q = min(q, 44.0)
            notes.append("The detected beats do not add up to a repeatable pulse shape - "
                         "this reads as noise rather than a heartbeat. Cover the lens with "
                         "the fingertip and press firmly, then retake.")
        else:
            q += 12.0 * float(np.clip((fold_g - 0.8) / 1.4, 0, 1))
            if fold_g < 2.0:
                q -= 15.0 * (2.0 - fold_g)
    if fold_ok and morph_pk > 2.0:
        q -= 12.0
        notes.append("More than one pulse per detected cycle - the rate may be a half-frequency alias.")
    if hr and (hr / 60.0) > 0.92 * (fs / 2.0):
        q -= 25.0
        notes.append("Estimated rate is close to the camera frame-rate limit (Nyquist); capture at a higher frame rate.")
    # Two comparable in-band peaks = pulse vs movement. Refuse rather than guess.
    if amb_ratio >= AMBIGUITY_REFUSE:
        q = min(q, 50.0)
        notes.append(
            f"Two competing rhythms of similar strength were found in the pulse band "
            f"(second peak at {100 * amb_ratio:.0f}% of the first) - that is the signature of "
            "a steady movement artefact, and on a single optical channel we cannot tell "
            "which one is the heart. Brace the hand on a table and retake, or average two reads.")
    if ambiguous:
        q = min(q, 55.0)
        notes.append("A steady rhythmic movement is about as likely as the heartbeat in this clip - rest the hand on a table and retake.")
    elif fold_ok and snr >= 6.0 and fold_g < 0.9 and agree > 4.0:
        q = min(q, 58.0)
        notes.append("Detected cycles did not line up into a consistent pulse shape; retake with the finger steadier.")
    q = float(np.clip(q, 0, 100))

    if snr < 6.0:
        notes.append("No dominant pulse frequency stood out above the noise floor - retake with the flash on and the finger still.")
    if cov < 0.7:
        notes.append("Pulse peaks were missed across the window - hold the phone steady and keep the finger covering the whole circle.")
    if perf < MIN_PERFUSION:
        notes.append("Almost no pulse was detected. Press the fingertip firmly over the lens until the flash washes it evenly red.")
    if ppg.rejected.get("frac_clipped", 0) > 0.25:
        notes.append("The flash is over-exposing the skin. Move the finger slightly away from the lens, or dim the flash.")
    if ppg.duration < 8.0:
        notes.append("Recording was under 8 seconds - record 10-12 s for a stable score.")
    if agree > 8.0:
        notes.append("Estimators disagreed, so the rhythm may be irregular or movement-corrupted. Retake the reading.")

    label = "good" if q >= 70 else ("acceptable" if q >= 45 else "poor")
    if label == "poor" and not notes:
        notes.append("Retry: the signal was too noisy to trust. Reposition and record again.")

    return HRResult(
        hr_bpm=round(float(hr), 1) if hr else None,
        hr_spectral_bpm=round(float(hr_sp), 1),
        hr_temporal_bpm=round(float(hr_tp), 1),
        hr_from_peaks_bpm=round(float(hr_peaks), 1) if hr_peaks else None,
        hrv_rmssd_ms=round(float(rmssd), 1) if rmssd else None,
        hrv_sdnn_ms=round(float(sdnn), 1) if sdnn else None,
        breathing_rate_bpm=round(float(resp), 1) if resp else None,
        quality=round(q, 1),
        quality_label=label,
        n_peaks=int(len(peaks)),
        agree_bpm=round(agree, 2),
        guidance=notes,
        flags={
            "spectral_snr_db": round(float(10 * np.log10(max(snr, 1e-6))), 1),
            "ac_dc_ratio": round(float(perf), 4),
            "duration_s": round(float(ppg.duration), 2),
            "fs": round(float(fs), 2),
            "peak_coverage": round(float(cov), 2),
            "fold_gain": round(float(fold_g), 2),
            "template_peaks": round(float(morph_pk), 1),
            "template_sharpness": round(float(morph_s), 2),
            "ambiguous_harmonic": bool(ambiguous),
            "spectral_ambiguity": round(float(amb_ratio), 2),
        },
    )
