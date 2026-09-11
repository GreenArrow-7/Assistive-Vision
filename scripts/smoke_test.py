"""Real inference smoke test on a generated sign; does not measure field accuracy."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from server import main

def run():
    main._warmup()
    client = TestClient(main.app)
    image = np.full((320, 800, 3), 255, np.uint8)
    cv2.putText(image, 'EXIT ROOM 205', (25,180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4)
    results = {}
    for name, frame in [('horizontal',image), ('tilted',cv2.warpAffine(image,
            cv2.getRotationMatrix2D((400,160),12,1),(800,320),borderValue=(255,255,255)))]:
        response = client.post('/api/search', files={'frame':('sign.jpg',cv2.imencode('.jpg',frame)[1].tobytes())},
                               data={'keyword':'room 205'})
        assert response.status_code == 200, response.text
        result = response.json()
        results[name] = result
        assert not result['component_errors'], result['component_errors']
        assert result['match'], f'{name} sign not recognized'
        assert result['match']['steps'] is None
        assert result['match']['quad']
    path = Path('runs/prototype-smoke.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({name:{'text':r['match']['label'],'ms':r['ms']} for name,r in results.items()}))

if __name__ == '__main__':
    run()
