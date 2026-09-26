"""Browser regressions for synthetic truth, rejection and capture lifecycle."""
import json,sys
from playwright.sync_api import sync_playwright
BASE=sys.argv[1] if len(sys.argv)>1 else 'http://localhost:8090'
with sync_playwright() as pw:
 b=pw.chromium.launch(args=['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'])
 p=b.new_page(viewport={'width':430,'height':900},is_mobile=True,has_touch=True)
 errors=[];requests=[]
 p.on('pageerror',lambda e:errors.append(str(e)))
 p.on('request',lambda r: requests.append(r.post_data_json) if r.url.endswith('/api/ppg/process') else None)
 p.goto(BASE,wait_until='networkidle')
 # Select a patient: demo must still neither associate nor save a reading.
 p.locator('#ptPick').evaluate("e=>{e.innerHTML+='<option value=1>Test patient</option>';e.value='1'}")
 for _ in range(5):
  p.locator('#btnSim').evaluate('el=>el.onclick()')
  assert abs(float(p.inner_text('#oHr'))-80)<3
  assert 'NOT your heart rate' in p.inner_text('#resultSource')
 assert all(r['source']=='simulation' and not r['save'] and r['patient_id'] is None for r in requests)
 assert len(requests)==5 # no silent re-rolls
 print('PASS: five measured 80 BPM demos, labelled synthetic, never saved, no re-roll')
 before=len(requests)
 p.evaluate('async()=>{const b=document.querySelector("#btnSim");await Promise.all([b.onclick(),b.onclick(),b.onclick()])}')
 assert len(requests)==before+1
 print('PASS: three overlapping invocations create exactly one request')
 p.check('#simHard')
 p.locator('#btnSim').evaluate('el=>el.onclick()')
 assert p.inner_text('#oHr')=='—'
 assert p.inner_text('#oRisk')=='not scored'
 assert p.locator('#oFactors .factor').count()==0
 print('PASS: shaky signal displays no measured HR, risk or SHAP factors')
 p.uncheck('#simHard')
 # Fault injection in the TEST browser only, not the app/backend.
 def wrong(route):
  response=route.fetch();data=response.json()
  data['heart_rate']['hr_bpm']=196
  data['heart_rate']['quality']=100
  data['accepted']=True;data['retry_required']=False
  route.fulfill(response=response,json=data)
 p.route('**/api/ppg/process',wrong)
 p.locator('#btnSim').evaluate('el=>el.onclick()')
 assert p.inner_text('#oHr')=='—' and p.inner_text('#oRisk')=='not scored'
 assert 'validation failed' in p.inner_text('#oAdvice').lower()
 print('PASS: injected 196 BPM / high-quality server answer refused for 80 BPM demo')
 p.unroute('**/api/ppg/process',wrong)
 p.click('#btnCam');p.wait_for_timeout(1800)
 assert p.locator('#video').is_visible()
 assert p.evaluate('window.__vg.simCanvas===null && !!window.__vg.stream')
 assert p.evaluate('window.__vg.roi.x>=0 && window.__vg.roi.y>=0')
 print('PASS: starting camera after demo restores visible video and camera ROI')
 p.evaluate('window.oldTrack=window.__vg.stream.getVideoTracks()[0]')
 p.locator('#btnSim').evaluate('el=>el.onclick()')
 assert p.evaluate('window.oldTrack.readyState==="ended" && window.__vg.stream===null')
 print('PASS: starting a demo stops physical-camera sampling')
 assert not errors,errors
 b.close()
 print('ALL SIGNAL-SAFETY BROWSER CHECKS PASSED')
