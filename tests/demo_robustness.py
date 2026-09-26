"""Demo-robustness check: click "Simulate fingertip" repeatedly and require that
every click ends with a scored result (never a stalled or errored panel), then
confirm "shaky finger" mode still gets refused.

    python tests/demo_robustness.py [base_url]
"""

import sys
import re

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
N = 12
# Wait on the status line, which is written LAST by runSim - waiting on the result
# fields races with the gated -> re-roll sequence and reads a transient state.
WAIT = ("(document.querySelector('#progTxt').textContent.indexOf('simulated truth') !== -1)"
        " || (document.querySelector('#progTxt').textContent.indexOf('was gated') !== -1)"
        " || (document.querySelector('#progTxt').textContent.indexOf('DEMO rejected') !== -1)"
        " || ((document.querySelector('#diag') || {style:{display:'none'}}).style.display === 'block')")


def main() -> int:
    fails = 0
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        p = b.new_context(viewport={"width": 414, "height": 896}, is_mobile=True, has_touch=True).new_page()
        errs: list[str] = []
        p.on("pageerror", lambda e: errs.append(str(e)))
        p.goto(BASE, wait_until="networkidle")

        scored = 0
        rows = []
        for _ in range(N):
            p.click("#btnSim")
            try:
                p.wait_for_function(WAIT, timeout=20_000)
                hr, risk, qual = p.inner_text("#oHr"), p.inner_text("#oRisk"), p.inner_text("#oQual")
                truth = re.search(r"ground truth ([\d.]+)", p.inner_text("#oNotes"))
                ok = bool(risk.strip() and risk.strip()[0].isdigit() and truth
                          and abs(float(hr) - float(truth.group(1))) <= 3)
                if not ok: fails += 1
                rows.append((hr, risk, qual, "ok" if ok else "gated"))
                scored += 1 if ok else 0
            except Exception:
                rows.append(("TIMEOUT", p.inner_text("#oRisk"), p.inner_text("#oQual"), "FAIL"))
                fails += 1
            p.wait_for_timeout(200)

        print(f"\n{N} rapid 'Simulate fingertip' clicks -> {scored}/{N} produced a scored result")
        for i, (hr, risk, q, tag) in enumerate(rows):
            print(f"   {i+1:2}: HR {hr:>7}  risk {risk:>7}  quality {q:>6}   {tag}")
        print("uncaught JS errors:", len(errs), errs[:2])
        if len(errs):
            fails += 1

        p.check("#simHard")
        p.click("#btnSim")
        try:
            p.wait_for_function(WAIT, timeout=20_000)
        except Exception:
            pass
        tier, prog = p.inner_text("#oTier"), p.inner_text("#progTxt")
        print(f"\nshaky-finger mode -> tier '{tier}' | status: {prog.strip()[:70]}")
        if "retry" not in tier.lower():
            print("   FAIL: shaky demo was not refused")
            fails += 1
        b.close()

    print("\n" + ("DEMO ROBUSTNESS OK" if fails == 0 else f"{fails} PROBLEM(S)"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
