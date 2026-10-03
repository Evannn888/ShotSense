"""Local DNG recommendations and experimental JPEG estimates in bounded workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import streamlit as st
import numpy as np
from src.preview import VERSION,render_linear_preview,validate_source

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'artifacts/model'
LABELS = {'Exposure':'Exposure', 'Contrast':'Contrast', 'Saturation':'Saturation',
          'Temperature':'Temperature', 'Tint':'Tint', 'HighlightRecovery':'Highlight recovery (legacy)'}


@st.cache_resource
def job_lock():
    # ponytail: one RAW job at a time; use a queue if remote multi-user support is needed.
    return threading.Lock()


def clear_prediction():
    st.session_state.pop('prediction', None)
    for key in ('jpeg_preview_estimates','jpeg_preview_manual','jpeg_manual_exposure','jpeg_manual_recovery'):
        st.session_state.pop(key,None)


def run_job(data,suffix='.dng'):
    started=time.perf_counter()
    if not 0 < len(data) <= 128 * 1024 * 1024:
        raise ValueError('The file must be nonempty and no larger than 128 MB.')
    suffix=suffix.lower()
    if suffix not in ('.dng','.jpg','.jpeg'): raise ValueError('Supported inputs are DNG, JPG and JPEG.')
    with job_lock(), tempfile.TemporaryDirectory(prefix='shotsense-') as directory:
        folder = Path(directory)
        source, result, preview = [folder / name for name in ('input'+suffix,'result.json','preview-source.npz')]
        source.write_bytes(data)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        module='src.inference' if suffix=='.dng' else 'src.jpeg_inference'
        process = subprocess.run([sys.executable, '-m', module, str(source),
                                  '--bundle', str(BUNDLE), '--output', str(result), '--preview-source', str(preview)],
                                 cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        if process.returncode:
            detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else 'Unable to decode the image.'
            if 'camera white balance' in detail:
                detail='This DNG lacks valid camera white balance information and is not supported by the current model.'
            elif '40 megapixel' in detail:
                detail='The image exceeds the 40-megapixel processing limit.'
            raise ValueError(detail)
        payload=json.loads(result.read_text())
        payload['timing']['local_worker_seconds']=time.perf_counter()-started
        with np.load(preview,allow_pickle=False) as cached:
            linear=validate_source(cached['linear_rgb'].copy())
        return payload, linear


st.set_page_config(page_title='ShotSense · Photo adjustment recommendations', page_icon='📷', layout='wide')
st.title('ShotSense')
st.caption('Preview version: ' + VERSION + ' · High-resolution preview with original aspect ratio')
st.write('Upload an unedited DNG for parameter recommendations, or a JPG/JPEG for experimental estimates.')
st.info('Research prototype · Recommendations are absolute values for legacy Camera Raw PV2003. Highlight recovery cannot be mapped directly to modern Highlights. Parameters that have not passed validation are listed separately as experimental outputs.')

if not (BUNDLE / 'model.json').exists():
    st.warning('The model bundle is unavailable. Restore the verified artifacts/model bundle or follow the training and acceptance instructions in README.md.')
    st.stop()

upload = st.file_uploader('Upload a DNG, JPG or JPEG', type=['dng','jpg','jpeg'], on_change=clear_prediction,
                          help='Maximum 128 MB and 40 megapixels. JPEG estimates are experimental. Processed locally; not uploaded to a remote server.')
left, right = st.columns(2)
analyze = left.button('Generate recommendations', type='primary', disabled=upload is None)
sample = right.button('Use project sample')
if analyze or sample:
    clear_prediction()
    try:
        with st.spinner('Reading the image, extracting features, and running inference…'):
            data = upload.getvalue() if analyze else next(iter(sorted((ROOT / 'data/raw/dngs').glob('*.dng')))).read_bytes()
            st.session_state.prediction = run_job(data,Path(upload.name).suffix if analyze else '.dng')
            st.session_state.prediction[0]['input_name']=upload.name if analyze else 'Project sample'
    except subprocess.TimeoutExpired:
        st.error('Processing exceeded 60 seconds and was stopped. Please use a smaller image.')
    except (ValueError, OSError, StopIteration) as error:
        st.error('Processing failed: ' + str(error))

if 'prediction' in st.session_state:
    result, preview = st.session_state.prediction
    is_jpeg=result.get('input_format')=='JPEG'
    if is_jpeg:
        st.warning('JPEG estimates are experimental: the model was trained on unedited RAW files. Temperature and tint are unavailable; clipped detail cannot be recovered.')
    if isinstance(preview,bytes):
        st.info('The preview has been upgraded. Generate recommendations again to get a high-resolution preview with the original aspect ratio.')
        st.stop()
    st.caption('Current result: ' + result.get('input_name','DNG'))
    image_column, parameter_column = st.columns([3, 2])
    with image_column:
        applied={name:value for name,value in result['recommended_absolute'].items()
                 if name in ('Exposure','HighlightRecovery')}
        preview_mode='model_recommendations'
        if is_jpeg:
            preview_mode='unchanged'
            if st.checkbox('Apply experimental JPEG estimates to preview',value=False,key='jpeg_preview_estimates'):
                applied={name:value for name,value in result['experimental_absolute'].items()
                         if name in ('Exposure','HighlightRecovery')}
                preview_mode='experimental_estimates'
            if st.checkbox('Adjust JPEG preview manually',value=False,key='jpeg_preview_manual'):
                applied={'Exposure':st.slider('Manual exposure (EV)',-4.0,4.0,0.0,step=0.1,key='jpeg_manual_exposure'),
                         'HighlightRecovery':st.slider('Manual highlight compression',0,100,0,key='jpeg_manual_recovery')}
                preview_mode='manual'
                st.caption('Manual values replace the experimental estimates for this preview.')
            if preview_mode=='unchanged':
                st.info('No JPEG adjustments are applied. Enable experimental estimates or manual adjustment above to change the preview.')
        strength=st.slider('Adjustment strength',0,100,100,step=5,key='preview_strength',disabled=is_jpeg and preview_mode=='unchanged')/100
        protect=st.checkbox('Protect highlights',value=True,key='preview_protect',disabled=is_jpeg and preview_mode=='unchanged')
        baseline,rendered,preview_metadata=render_linear_preview(preview,applied,strength,protect)
        preview_metadata['adjustment_mode']=preview_mode
        unchanged=baseline==rendered
        if not (is_jpeg and preview_mode=='unchanged') and unchanged:
            st.info('The current settings produce no visible change.' + (' Use manual exposure to adjust brightness.' if is_jpeg else ''))
        if is_jpeg:
            preview_metadata.update(source='Linearized rendered sRGB JPEG after EXIF orientation and ICC handling',
                source_ceiling_semantics='Rendered JPEG display ceiling; not sensor exposure. Missing detail cannot be reconstructed.',
                semantics='Custom global tone preview on an already processed JPEG; optional experimental estimates or manual values, not RAW/Lightroom equivalence.')
        before,after=st.columns(2)
        with before:
            st.image(baseline, caption='Before · Decoded JPEG' if is_jpeg else 'Before · Baseline RAW development', width='stretch')
        with after:
            st.image(rendered, caption='Unchanged · No visible adjustment' if unchanged else 'Approximate result · Exposure and highlight compression', width='stretch')
        status='JPEG estimates apply only when enabled above.' if is_jpeg else 'Experimental parameters are excluded.'
        st.caption(f"Approximate preview ({preview_metadata['size'][0]} × {preview_metadata['size'][1]}): Original aspect ratio; only exposure and highlight compression are simulated. {status} This is not equivalent to Lightroom and cannot guarantee recovery of clipped detail.")
        effective=preview_metadata['applied_parameters']
        st.caption(f"Applied preview: Exposure {effective['Exposure']:.2f} EV · Highlight compression proxy {effective['HighlightRecovery']:.2f} · New pixels with a full-value channel {preview_metadata['new_full_channel_fraction']:.2%}")
        st.caption(f"Baseline preview at ceiling: At least one channel {preview_metadata['source_channel_ceiling_fraction']:.2%} · All three channels white {preview_metadata['source_white_ceiling_fraction']:.2%}")
        if preview_metadata['source_channel_ceiling_fraction'] >= .001:
            detail='The rendered JPEG already has channels at the ceiling.' if is_jpeg else 'The baseline preview already has channels at the ceiling, possibly from RAW development or display gamut clipping.'
            st.warning(detail+' Dimming these regions does not restore texture; fully white regions may still look flat.')
        if not protect:
            st.warning('Highlight protection is off; direct exposure may introduce clipping.')
        st.download_button('Download approximate preview PNG', rendered, file_name='shotsense-approximate-preview.png',mime='image/png')
    with parameter_column:
        st.subheader('Parameter recommendations')
        if result['recommended_absolute']:
            for name, value in result['recommended_absolute'].items():
                st.metric(LABELS[name], f"{value:.2f} {result['units'][name]}")
            st.caption('These parameters outperform the training-median baseline on the fixed validation set. Acceptable error for practical use has not been calibrated.')
        else:
            st.warning('No parameters are validated for JPEG input. Estimates appear below.' if is_jpeg else 'No parameters have passed validation for this model. All outputs are experimental.')
        with st.expander('Experimental outputs',expanded=is_jpeg):
            for name, value in result['experimental_absolute'].items():
                st.write(f"{LABELS[name]}: {value:.2f} {result['units'][name]}")
            if is_jpeg: st.caption('Temperature and tint: unavailable for JPEG input.')
        export_payload=dict(result,preview=preview_metadata)
        st.download_button('Download parameters JSON', json.dumps(export_payload, ensure_ascii=False, indent=2),
                           file_name='shotsense-parameters.json', mime='application/json')
        st.caption(f"Local job {result['timing']['local_worker_seconds']:.2f} s · Image to estimates {result['timing']['end_to_end_seconds']:.2f} s · Preview source {result['timing'].get('preview_source_seconds',0):.2f} s · Model {result['timing']['model_forward_ms']:.1f} ms")
        st.caption('Model version: ' + result['model_sha256'][:16])
