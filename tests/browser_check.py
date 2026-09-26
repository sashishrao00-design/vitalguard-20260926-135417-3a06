"""Real-browser verification of the client (Playwright + headless Chromium).

    pip install playwright && python -m playwright install chromium
    python tests/browser_check.py [base_url]

Loads the page, fails on ANY uncaught JS error, then drives the buttons:
simulate -> result panel, ward -> table, alerts, model. With
--use-fake-device-for-media-stream it also opens the "camera", so the real
getUserMedia -> ROI readback -> POST path executes exactly as it does on a phone.
"""

from __future__ import annotations

import re
import sys
import time

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
ARGS = [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
]

fails: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(f"{name}: {detail}")


def main() -> int:
    js_errors: list[str] = []
    console_errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=ARGS)
        ctx = browser.new_context(viewport={"width": 430, "height": 900},    # phone-shaped
                                  is_mobile=True, has_touch=True)
        page = ctx.new_page()
        page.on("pageerror", lambda e: js_errors.append(str(e)))
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)

        print(f"\n=== {BASE} (mobile viewport 430x900) ===")
        resp = page.goto(BASE, wait_until="networkidle", timeout=45_000)
        check("page loads", resp.status == 200, f"HTTP {resp.status}")
        check("styles applied", page.evaluate(
            "getComputedStyle(document.querySelector('.card')).borderRadius") != "0px", "card radius present")
        check("no rediag banner", page.locator("#diag").count() == 0 or
              "blocked" not in (page.locator("#diag").inner_text() if page.locator("#diag").count() else "").lower())

        # ---- simulated capture (no camera permission needed) ----
        page.click("#btnSim")
        try:
            page.wait_for_selector("#resultBody:not([hidden])", timeout=25_000)
            ok = True
        except Exception:
            ok = False
        hr = page.inner_text("#oHr") if ok else "—"
        risk = page.inner_text("#oRisk") if ok else "—"
        qual = page.inner_text("#oQual") if ok else "—"
        factors = page.locator("#oFactors .factor").count() if ok else 0
        check("Simulate fingertip produced a result", ok, f"HR={hr} q={qual} risk={risk}")
        check("heart rate is a plausible number", bool(re.fullmatch(r"\d{2,3}(\.\d)?", hr.strip())), f"got {hr!r}")
        # "6%" and "6.9%" are both valid: the server rounds to 1 dp, so a whole
        # number renders without the decimal.
        check("risk percentage rendered", bool(re.fullmatch(r"\d{1,3}(\.\d)?%", risk.strip())), f"got {risk!r}")
        check("SHAP factor bars rendered", factors >= 2, f"{factors} factors")
        prog = page.inner_text("#progTxt")
        check("progress line reports the run", "simulated truth" in prog or "frames" in prog, prog.strip()[:70])

        # ---- real camera path via fake device ----
        page.click("#btnCam")
        page.wait_for_timeout(1500)
        streaming = page.evaluate("!!document.querySelector('video').srcObject")
        check("getUserMedia started a stream", streaming)
        page.wait_for_timeout(1500)
        st = page.evaluate("""() => { const S = window.__vg || {};
              return { n: (S.frames||[]).length, vw: document.querySelector('video').videoWidth,
                       roi: S.roi || null, useVfc: !!S.useVfc, err: S.warned || false }; }""")
        check("camera frame geometry is measurable", st["vw"] > 0, f"videoWidth={st['vw']}")
        check("ROI box computed (non-zero)", bool(st["roi"]) and st["roi"]["s"] >= 8, str(st["roi"]))
        check("frames sampled from the live camera", st["n"] > 5, f"{st['n']} frames, rVFC={st['useVfc']}")
        check("no per-frame sampling errors", not st["err"])

        # short-record attempt: proves the recorder runs; the fake device has no pulse,
        # so the correct behaviour is either a score or an explicit refusal - never a crash
        page.evaluate("""() => { window.__rec = true; }""")
        page.click("#btnRec")
        page.wait_for_timeout(13_500)
        check("record completed without throwing", not js_errors, "; ".join(js_errors[:2]))
        msg = page.inner_text("#camMsg") + " " + page.inner_text("#resultEmpty" if page.locator("#resultEmpty").is_visible() else "#oNotes")
        notes.append(msg.strip().replace("\n", " ")[:160])

        # ---- other tabs ----
        for tab, sel, want in [("ward", "#ptBody tr", 3), ("alerts", "#alertList .alert", 1), ("model", "#metricsBody .box", 4)]:
            page.click(f'.tab[data-tab="{tab}"]')
            page.wait_for_timeout(1200)
            got = page.locator(sel).count()
            check(f"{tab} tab renders", got >= want, f"{got} elements (wanted >={want})")

        # ---- PPG plot + gauge actually drew pixels ----
        page.click('.tab[data-tab="capture"]')
        page.wait_for_timeout(600)
        drew = page.evaluate("""() => {
          const c = document.querySelector('#ppgPlot'); const x = c.getContext('2d');
          const d = x.getImageData(0,0,c.width,c.height).data; let lit=0;
          for (let i=3;i<d.length;i+=4) if (d[i]>0) lit++; return lit; }""")
        check("PPG waveform canvas has pixels", drew > 500, f"{drew} lit samples")
        gauge = page.evaluate("""() => { const c=document.querySelector('#gauge');
          const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data; let lit=0;
          for(let i=3;i<d.length;i+=4) if(d[i]>0) lit++; return lit; }""")
        check("risk gauge canvas has pixels", gauge > 300, f"{gauge} lit samples")

        check("no uncaught JS errors", not js_errors, " | ".join(js_errors[:3]) or "clean")
        real_console = [c for c in console_errors if "favicon" not in c and "Manifest" not in c]
        check("no console errors", not real_console, " | ".join(real_console[:2]) or "clean")

        # ---- landing page + QR ----
        page.goto(BASE + "/phone", wait_until="domcontentloaded")
        check("/phone landing page renders", page.locator("img[alt*='QR']").count() == 1)
        href = page.locator("a.u").get_attribute("href") or ""
        check("/phone link matches the serving origin", href.startswith(BASE), href)
        check("QR image is embedded", "data:image/png;base64" in (page.content() or ""))

        browser.close()

    print("\n  capture/refusal text seen:", notes[-1] if notes else "(n/a)")
    print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILURE(S)'}")
    for f in fails:
        print("   -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
