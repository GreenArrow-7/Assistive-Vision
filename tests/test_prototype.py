import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from server import main, spatial, symbols, text_pipeline
from server.navigation import MapsHandoffProvider

def test_no_floor_distance_for_text_or_stairs():
    for label in ('EXIT', 'chair', 'WASHROOM'):
        item = {'label': label, 'kind': 'text', 'box': (10, 500, 500, 710)}
        spatial.annotate([item], 960, 720)
        assert item['steps'] is None and not item['proximity']
    assert spatial.estimate_steps('stairs_down', (20, 400, 900, 710), 960, 720) is None

@pytest.mark.parametrize('box,w,h', [((0,0,0,0),960,720), ((0,0,10,10),0,720), ((0,0,float('nan'),10),960,720)])
def test_invalid_geometry(box,w,h):
    assert spatial.estimate_steps('chair',box,w,h) is None

@pytest.mark.parametrize('query,text,expected', [
    ('men','menu',False), ('men','women',False), ('room 205','ROOM 206',False),
    ('room 205','room',False), ('room 205','ROOM: 205',True),
    ('pharmacy','pharrnacy',False), ('pharmacy','pharm acy',False),
    ('pharmacy','pharnacy',True), ('fire exit','FIRE   EXIT',True)])
def test_safe_matching(query,text,expected):
    assert symbols.safe_match(text,query) is expected

def test_rank_nearest_match():
    a={'label':'EXIT','steps':8,'conf':.95}
    b={'label':'EXIT','steps':3,'conf':.9}
    assert symbols.match_keyword('exit',[a,b],[],[]) is b

def test_navigation_fallback_and_validation():
    client=TestClient(main.app)
    r=client.post('/api/navigation',json={'destination':'Bangalore','latitude':12.2,'longitude':76.1})
    assert r.status_code == 200
    assert r.json()['distance_m'] is None and r.json()['resolved'] is False
    assert 'origin=' in r.json()['maps_url']
    assert 'origin=' not in MapsHandoffProvider().route('Bangalore')['maps_url']
    for body in ({'destination':' '},{'destination':'A','latitude':100,'longitude':0},{'destination':'A','latitude':1}):
        assert client.post('/api/navigation',json=body).status_code == 422

def test_partial_failure_keeps_objects(monkeypatch):
    monkeypatch.setitem(main._state,'ready',True)
    main._rate.clear()
    monkeypatch.setattr(main.object_provider,'detect',lambda _:([{'label':'chair','kind':'object','box':(10,10,40,50),'conf':.8}],[]))
    def fail(_): raise RuntimeError('private model path')
    monkeypatch.setattr(main.text_provider,'detect',fail)
    image=np.random.default_rng(5).integers(0,255,(100,100,3),dtype=np.uint8)
    r=TestClient(main.app).post('/api/analyze/frame',files={'frame':('frame.jpg',cv2.imencode('.jpg',image)[1].tobytes())})
    assert r.status_code == 200
    assert r.json()['objects'] and r.json()['component_errors']
    assert 'private model path' not in r.text

def test_oriented_quad_preserved(monkeypatch):
    quad=[[10,20],[90,10],[95,30],[15,40]]
    class Reader:
        def readtext(self,*a,**k): return [(quad,'EXIT',.95)]
    monkeypatch.setattr(text_pipeline,'_get_obb',lambda:None)
    monkeypatch.setattr(text_pipeline,'_get_reader',lambda:Reader())
    result=text_pipeline.detect_text(np.zeros((100,100,3),dtype=np.uint8))
    assert result[0]['quad']==quad and result[0]['angle'] != 0

def test_expired_track_not_confirmed(monkeypatch):
    monkeypatch.setattr(main.time,'time',lambda:1000)
    main._sessions['expired']={'ts':0,'items':[('chair',(0,0,20,20))]}
    assert main._confirm('expired',[{'label':'chair','box':(0,0,20,20)}]) == []

def test_adjacent_words_form_phrase_but_separate_rows_do_not():
    def word(label,x,y):
        return {'label':label,'quad':[[x,y],[x+40,y],[x+40,y+20],[x,y+20]],
                'box':(x,y,x+40,y+20),'angle':0,'conf':.9,'kind':'text'}
    joined=text_pipeline.merge_text_lines([word('205',50,0),word('ROOM',0,0),word('206',50,40)])
    assert {i['label'] for i in joined} == {'ROOM 205','206'}

def test_chunked_body_is_bounded_without_content_length():
    main._rate.clear()
    def chunks():
        for _ in range(9):
            yield b'x' * (1024*1024)
    response=TestClient(main.app).post('/api/analyze/frame', content=chunks(),
        headers={'Content-Type':'application/octet-stream'})
    assert response.status_code == 413
