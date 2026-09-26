#!/usr/bin/env python3
"""Render a scannable PNG for a URL and prove it decodes back to that exact string.

Shared by phone_link.sh (cloudflared) and lhr_link.sh (localhost.run) so both providers
hand out a QR that actually matches whatever URL is live right now.
"""
import sys


def main() -> int:
    url, out = sys.argv[1], sys.argv[2]
    caption = sys.argv[3] if len(sys.argv) > 3 else "Scan with your phone"
    import qrcode.constants
    from qrcode import QRCode
    from PIL import Image, ImageDraw, ImageFont

    q = QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=12, border=2)
    q.add_data(url); q.make(fit=True)
    img = q.make_image(fill_color="#0b1220", back_color="white").convert("RGB")
    w, h = img.size; pad, ch = 28, 78
    cv = Image.new("RGB", (w + pad * 2, h + pad * 2 + ch), "#0d1524"); cv.paste(img, (pad, pad))
    d = ImageDraw.Draw(cv)
    try:
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        fb = fs = ImageFont.load_default()
    d.text((cv.width / 2, h + pad + 20), caption, fill="#e6efff", font=fb, anchor="mm")
    d.text((cv.width / 2, h + pad + 52), url.replace("https://", ""), fill="#8ea3c4", font=fs, anchor="mm")
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True); cv.save(out)

    # A QR printed for a dead tunnel is worse than no QR at all, so verify it.
    try:
        import cv2, numpy as np
        text, _, _ = cv2.QRCodeDetector().detectAndDecode(np.array(cv2.imread(out)))
        status = "scans OK" if text == url else f"MISMATCH decoded={text!r}"
    except Exception as exc:                                    # pragma: no cover
        status = f"(decode check unavailable: {type(exc).__name__})"
    print(f"  wrote {out}  -> {status}")
    return 0 if status == "scans OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
