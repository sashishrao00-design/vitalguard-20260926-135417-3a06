"""Regression checks for source attribution and refusal before risk/scoring."""
import math
import pytest
from fastapi.testclient import TestClient
from vitalguard import server
from vitalguard.ppg import synthetic_recording
from vitalguard.heart_rate import HRResult


def payload(source='simulation', truth=80):
    t,v=synthetic_recording(hr_bpm=80,seconds=12,fps=20,snr_db=18,seed=9)
    return {'frames':[{'t':float(a),'v':float(b)} for a,b in zip(t,v)],
            'source':source,'expected_hr_bpm':truth,'save':False,'vitals':{}}


def estimate(hr=80, quality=100):
    return HRResult(hr,hr,hr,hr,10,10,15,quality,'excellent',16,0,[],{'spectral_snr_db':20})


def test_wrong_demo_is_not_scored_or_displayed(monkeypatch):
    monkeypatch.setattr(server,'analyze',lambda _:estimate(196))
    monkeypatch.setattr(server,'score_vitals',lambda _:pytest.fail('mismatch was scored'))
    j=TestClient(server.app).post('/api/ppg/process',json=payload()).json()
    assert j['source']=='simulation'
    assert not j['accepted'] and j['retry_required'] and j['risk'] is None
    assert j['heart_rate']['hr_bpm'] is None
    assert j['heart_rate']['flags']['candidate_hr_bpm']==196
    assert j['simulation_check']['error_bpm']==116


def test_demo_never_saved_even_if_caller_requests_it(monkeypatch):
    monkeypatch.setattr(server,'analyze',lambda _:estimate())
    monkeypatch.setattr(server,'score_vitals',lambda _: (0.1, {'tier':'low'}))
    monkeypatch.setattr(server.db,'add_reading',lambda *a,**k:pytest.fail('synthetic data saved'))
    body=payload();body.update(save=True,patient_id=1)
    j=TestClient(server.app).post('/api/ppg/process',json=body).json()
    assert j['accepted'] and j['heart_rate']['hr_bpm']==80
    assert 'saved_reading_id' not in j


def test_refused_camera_does_not_display_hr(monkeypatch):
    monkeypatch.setattr(server,'analyze',lambda _:estimate(196,20))
    monkeypatch.setattr(server,'score_vitals',lambda _:pytest.fail('poor quality scored'))
    j=TestClient(server.app).post('/api/ppg/process',json=payload('camera',None)).json()
    assert j['heart_rate']['hr_bpm'] is None and j['risk'] is None
    assert j['heart_rate']['hrv_rmssd_ms'] is None


def test_camera_estimates_are_not_forced_to_demo_range(monkeypatch):
    monkeypatch.setattr(server,'analyze',lambda _:estimate(196,100))
    monkeypatch.setattr(server,'score_vitals',lambda _: (0.9, {'tier':'critical'}))
    j=TestClient(server.app).post('/api/ppg/process',json=payload('camera',None)).json()
    assert j['accepted'] and j['heart_rate']['hr_bpm']==196
    assert 'simulation_check' not in j


def test_simulation_requires_truth_and_camera_rejects_truth():
    client=TestClient(server.app)
    assert client.post('/api/ppg/process',json=payload('simulation',None)).status_code==422
    assert client.post('/api/ppg/process',json=payload('camera',80)).status_code==422


def test_frames_reject_nonfinite():
    with pytest.raises(ValueError):server.Frame(t=0,v=math.nan)
    with pytest.raises(ValueError):server.Frame(t=math.inf,v=130)


def test_unchanged_estimator_recovers_known_demo_signal():
    j=TestClient(server.app).post('/api/ppg/process',json=payload()).json()
    assert j['accepted'] and abs(j['heart_rate']['hr_bpm']-80)<2
    assert j['simulation_check']['passed']
