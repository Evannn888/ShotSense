from pathlib import Path
import pytest
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_local_app_sample_inference_and_download():
    if not (ROOT/'artifacts/model/model.json').exists():
        pytest.skip('Complete full training and export before application acceptance')
    at=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
    assert not at.exception
    next(button for button in at.button if button.label=='Use project sample').click().run(timeout=60)
    assert not at.exception and not at.error
    result,preview=at.session_state['prediction']
    assert preview.shape[2]==3 and max(preview.shape[:2])==1600 and preview.shape[0]!=preview.shape[1]
    assert len(result['recommended_absolute'])+len(result['experimental_absolute'])==6
    assert len(at.get('imgs'))==2
    downloads=at.get('download_button')
    assert {item.proto.label for item in downloads}=={'Download approximate preview PNG','Download parameters JSON'}
    original=result['recommended_absolute'].copy()
    at.slider[0].set_value(50).run(timeout=20)
    assert not at.exception and not at.error
    assert at.session_state['prediction'][0]['recommended_absolute']==original
    assert any('Applied preview: Exposure 0.40 EV' in caption.value for caption in at.caption)
    at.checkbox[0].uncheck().run(timeout=20)
    assert not at.exception and any('Highlight protection is off' in warning.value for warning in at.warning)

    at.session_state['prediction']=(result,np.ones((10,20,3),dtype=np.float32))
    at.run(timeout=20)
    assert not at.exception
    assert any('does not restore texture' in warning.value for warning in at.warning)
    assert any('All three channels white 100.00%' in caption.value for caption in at.caption)
