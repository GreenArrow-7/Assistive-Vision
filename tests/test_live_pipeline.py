import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from fastapi.testclient import TestClient
from server import main, priority, text_pipeline

def jpeg():
    # Low-texture image deliberately exercises the former blur hard-stop.
    return cv2.imencode('.jpg', np.full((120,160,3),35,np.uint8))[1].tobytes()

def test_blur_does_not_disable_detection_and_no_camera_steering_prompt(monkeypatch):
    monkeypatch.setitem(main._state,'ready',True)
    main._rate.clear();calls=[]
    monkeypatch.setattr(main.object_provider,'detect',lambda im:(calls.append('objects') or [],[]))
    monkeypatch.setattr(main.text_provider,'detect',lambda im:calls.append('text') or [])
    response=TestClient(main.app).post('/analyze',files={'frame':('f.jpg',jpeg())})
    assert response.status_code==200
    assert calls==['objects','text']
    assert response.json()['frame_quality']=='limited'
    assert response.json()['blur'] is False
    assert not any(word in response.json()['speech'].lower() for word in ('move','steady','point'))

def test_objects_finish_while_ocr_is_blocked_and_no_ocr_backlog(monkeypatch):
    monkeypatch.setitem(main._state,'ready',True)
    main._rate.clear();entered=threading.Event();release=threading.Event()
    def slow_ocr(im):
        entered.set();assert release.wait(5);return []
    monkeypatch.setattr(main.text_provider,'detect',slow_ocr)
    monkeypatch.setattr(main.object_provider,'detect',lambda im:([],[]))
    def request(path):
        return TestClient(main.app).post(path,files={'frame':('f.jpg',jpeg())})
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=pool.submit(request,'/api/detect/text')
        try:
            assert entered.wait(2)
            fast=pool.submit(request,'/api/detect/objects').result(timeout=2)
            assert fast.status_code==200 and fast.json()['object_active']
            assert fast.json()['ocr_active'] is False
            busy=request('/api/detect/text')
            assert busy.status_code==503 and busy.json()['busy']
        finally:release.set()
        assert pending.result().status_code==200

def test_high_confidence_immediate_hazard_bypasses_confirmation(monkeypatch):
    monkeypatch.setitem(main._state,'ready',True);main._rate.clear();main._sessions.clear()
    monkeypatch.setattr(main.object_provider,'detect',lambda im:([],[{
        'label':'chair','raw':'chair','box':(0,0,155,119),'conf':.95,'kind':'hazard'}]))
    response=TestClient(main.app).post('/api/detect/objects',files={'frame':('f.jpg',jpeg())},
                                      data={'mode':'live','session':'critical-test'})
    assert response.status_code==200
    assert response.json()['hazards'][0]['critical']
    assert response.json()['priority']==0 and response.json()['speech'].startswith('Warning!')
    assert not priority.is_critical({'raw':'chair','steps':1,'conf':.55})

def test_ocr_garbage_filter_retains_signs_and_numbers():
    for text in ['EXIT','ROOM 101','WC','101','Emergency exit']:
        assert text_pipeline.useful_text(text)
    for text in ['', '!!!???','aaaaaaa', 'a', '| | / /']:
        assert not text_pipeline.useful_text(text)
