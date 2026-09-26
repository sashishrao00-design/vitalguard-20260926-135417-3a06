# VitalGuard: host the app, not a temporary tunnel

## Status
This is a deployment-ready source package, **not an already deployed website**.
The application tests passed in the development workspace. The Docker build and
hosting-provider deployment have not been executed here.

## Deploy using GitHub and Render
1. Extract this ZIP on a computer.
2. Create a GitHub repository and upload the contents of the `vitalguard` folder.
   `Dockerfile`, `requirements.txt` and `train.py` must be at the repository root.
   Do not upload patient databases, account credentials or SSH keys.
3. Sign in at https://dashboard.render.com/ using your own account.
4. Choose **New → Web Service**, connect that repository, and use the **Docker**
   runtime. The included Dockerfile installs dependencies and trains the synthetic
   demonstration model during the build.
5. Configure the health-check path as `/api/health`. If selecting resources,
   allow at least 1 GB RAM as a starting point (not a measured minimum).
   Review the service's current pricing before confirming any paid plan.
6. Deploy and wait for the service to become live. Open the HTTPS service URL
   shown by the provider, then `/#capture`, on your phone.
7. Tap **Simulate fingertip** first. Then use **Start camera**, grant permission,
   and record a still fingertip. HTTPS alone does not guarantee torch support;
   camera and torch availability depend on the phone/browser.

This address is independent of the chat workspace. Availability still depends
on your hosting plan; some plans sleep when idle. Provider UI and plans may change.
If a build fails, share the error log, not your password or account token.

## Demo data and limits
- This prototype has **no login or access controls**. Use fictional data only.
- SQLite lives at `/app/data/vitalguard.db`. Without persistent storage, readings
  can disappear on redeployment/replacement. If supported by your chosen plan,
  mount a persistent disk at `/app/data` and run only one app instance.
- The ward initially has no patients. Use the API docs at `/docs` and execute
  `POST /api/seed` with `{"n":14,"seed":5}` to create fictional demo patients.
- The browser includes a doctor/ward view at `/#ward`. The separate Streamlit
  dashboard is included in the source but is not launched by this Dockerfile.
- The risk model is trained on synthetic labels, not clinical outcome data.
  Camera PPG estimates heart rate only; SpO2/BP/temp must be entered separately.
  Not a medical device and not suitable for clinical decisions.

## Local Docker alternative
```sh
docker build -t vitalguard .
docker run --rm -p 8000:8000 -v vitalguard-data:/app/data vitalguard
```
Open http://localhost:8000 on that computer. On an Android phone connected via
USB debugging, `adb reverse tcp:8000 tcp:8000` allows opening
http://localhost:8000 on the phone. A normal LAN HTTP URL generally cannot use
the browser camera; use HTTPS for remote phone capture.
