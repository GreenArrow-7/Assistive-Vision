import pytest
from scripts.evaluate_text import rotated_iou, edit_distance, evaluate

def test_rotated_iou():
    q=[[0,5],[5,0],[10,5],[5,10]]
    assert rotated_iou(q,q)==pytest.approx(1)
    assert rotated_iou(q,[[30,5],[35,0],[40,5],[35,10]])==0
    with pytest.raises(ValueError): rotated_iou([],q)

def test_recognition_and_missed_text():
    q=[[0,0],[10,0],[10,10],[0,10]]
    result=evaluate([{'truth':[{'quad':q,'text':'EXIT'}],
                      'predictions':[{'quad':q,'text':'EX1T'}]},
                     {'truth':[{'quad':q,'text':'ROOM'}], 'predictions':[]}])
    assert result['recall']==.5 and result['matched_cer']==.25
    assert result['matched_ocr_accuracy']==0
    assert edit_distance('men','menu')==1
