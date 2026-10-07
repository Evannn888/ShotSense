"""Local RAW recommendations, JPEG enhancement, and experimental estimates in bounded workers."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
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
PHOTO_RECIPES = {'Automatic conservative choice':'auto','Gentle lift':'natural','Gentle lift + chroma denoising':'chroma','Lift shadows':'shadows'}


@st.cache_resource
def job_lock():
    # ponytail: one RAW job at a time; use a queue if remote multi-user support is needed.
    return threading.Lock()


def clear_prediction():
    st.session_state.pop('prediction', None)
    for key in ('automatic_attempt','automatic_error','jpeg_preview_estimates','jpeg_preview_manual','jpeg_manual_exposure','jpeg_manual_recovery','restoration_strength','photo_strength','photo_shadow_strength','photo_shadow_result','photo_export_token','photo_saved_png','photo_inspect_details','photo_detail_x','photo_detail_y'):
        st.session_state.pop(key,None)
    reset_fine_tuning()


def reset_fine_tuning():
    for key in ('photo_fine_result','photo_warmth','photo_tint','photo_vibrance','photo_local_contrast','photo_export_token','photo_saved_png'):
        st.session_state.pop(key,None)


def show_png(data, caption, width='content'):
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Expected PNG image bytes')
    # A data URL preserves PNG pixels/profile; st.image(bytes) otherwise resizes and recompresses RGB to JPEG.
    st.image('data:image/png;base64,'+base64.b64encode(data).decode('ascii'),caption=caption,width=width)


def save_png_file(data, filename):
    if not data.startswith(b'\x89PNG\r\n\x1a\n') or Path(filename).name!=filename or not filename.endswith('.png'):
        raise ValueError('Expected PNG bytes and a plain .png filename')
    directory=Path.home()/'Downloads'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/filename
    try:
        with path.open('xb') as file: file.write(data)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes()!=data:
            raise FileExistsError('A different file already exists at '+str(path))
    return path


def local_png_button(data, filename, label='Save PNG to Downloads'):
    digest=hashlib.sha256(data).hexdigest()
    if st.button(label,key='local_png_save',type='primary',width='stretch'):
        st.session_state.pop('photo_saved_png',None)
        try:
            with st.spinner('Saving PNG locally…'):
                path=save_png_file(data,filename)
            st.session_state['photo_saved_png']=(digest,str(path))
        except (OSError,ValueError) as error:
            st.error('PNG was not saved: '+str(error))
    saved=st.session_state.get('photo_saved_png')
    if saved and saved[0]==digest:
        st.success('PNG saved to '+saved[1])


def prepare_png_download(token):
    st.session_state['photo_export_token']=token


def png_downloads(data, payload, png_label='Download full-resolution PNG', json_label='Download adjustment JSON'):
    field='export' if 'export' in payload else 'preview'
    metadata=dict(payload[field],output_png_sha256=hashlib.sha256(data).hexdigest())
    export_json=json.dumps(dict(payload,**{field:metadata},export_ui_version='png-local-delivery-v4'),ensure_ascii=False,indent=2)
    token=hashlib.sha256(export_json.encode()).hexdigest()
    w,h=metadata['size'];strength=metadata['strength']
    tuned='-finetuned' if payload.get('fine_tuning',{}).get('engine_applied') else ''
    name=f"shotsense-{payload.get('selected_recipe','preview')}{tuned}-s{strength*100:.0f}-{w}x{h}-{metadata['output_png_sha256'][:8]}"
    st.subheader('3 · Save your photo')
    st.caption(f'PNG · {w} × {h} · Saves the current result to this Mac.')
    local_png_button(data,name+'.png')
    with st.expander('Other download options',expanded=st.session_state.get('photo_export_token')==token):
        st.caption('Use browser download if you prefer to choose where the file is saved.')
        st.button('Prepare PNG download',on_click=prepare_png_download,args=(token,))
        if st.session_state.get('photo_export_token')==token:
            st.download_button(png_label,data,file_name=name+'.png',mime='image/png',
                               key='photo_png_'+token,on_click='ignore',width='stretch')
            with st.expander('Adjustment metadata (JSON)'):
                st.caption('JSON contains adjustment settings. Download the PNG to save the photo.')
                st.download_button(json_label,export_json,file_name=name+'.json',mime='application/json',
                                   key='photo_json_'+token,on_click='ignore')
        else:
            st.caption('Prepare the current PNG once. After changing adjustments, prepare it again.')


def choose_recipe(is_sample, input_hash, choice='Gentle lift'):
    clear_prediction()
    st.session_state['photo_workflow_v2']='Natural adjustment (experimental)'
    st.session_state['photo_recipe']=choice
    st.session_state['pending_recipe']=(is_sample,input_hash)


def run_worker(command):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,start_new_session=True) as process:
        try:
            _,stderr=process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.communicate()
            raise
    if process.returncode:
        raise ValueError(stderr.strip().splitlines()[-1] if stderr.strip() else 'Unable to process the image.')


def run_fine_tuning(rgb,settings):
    from src.photo_engine import MAX_JPEG_PIXELS,validate_fine_tuning
    settings=validate_fine_tuning(settings);started=time.perf_counter()
    if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3
            or min(rgb.shape[:2])<1 or rgb.shape[0]*rgb.shape[1]>MAX_JPEG_PIXELS):
        raise ValueError('Invalid fine-tuning source pixels')
    with job_lock(),tempfile.TemporaryDirectory(prefix='shotsense-fine-worker-') as directory:
        folder=Path(directory);source=folder/'source.npy';result=folder/'result.json';preview=folder/'pixels.npz'
        np.save(source,rgb,allow_pickle=False)
        run_worker([sys.executable,'-m','src.photo_engine',str(source),'--fine-tune',json.dumps(settings),
                    '--output',str(result),'--preview-source',str(preview)])
        metadata=json.loads(result.read_text())
        with np.load(preview,allow_pickle=False) as cached:
            adjusted=cached['adjusted'].copy()
        if adjusted.dtype!=np.uint8 or adjusted.shape!=rgb.shape:
            raise ValueError('Fine tuning returned invalid pixels')
        metadata['local_worker_seconds']=time.perf_counter()-started
        return metadata,adjusted


def run_job(data,suffix='.dng',enhance=False,natural=False,recipe='auto',color=False,scene_aware=False):
    started=time.perf_counter()
    if not 0 < len(data) <= 128 * 1024 * 1024:
        raise ValueError('The file must be nonempty and no larger than 128 MB.')
    suffix=suffix.lower()
    if suffix not in ('.dng','.jpg','.jpeg'): raise ValueError('Supported inputs are DNG, JPG and JPEG.')
    if color and suffix=='.dng': raise ValueError('Natural color currently supports JPG and JPEG only.')
    if scene_aware and not color: raise ValueError('Scene-aware adjustment currently supports the JPEG color workflow only.')
    with job_lock(), tempfile.TemporaryDirectory(prefix='shotsense-') as directory:
        folder = Path(directory)
        source, result, preview = [folder / name for name in ('input'+suffix,'result.json','preview-source.npz')]
        source.write_bytes(data)
        restoration = enhance and suffix != '.dng' and not natural and not color
        module='src.lut_natural' if color else 'src.photo_engine' if natural else ('src.inference' if suffix=='.dng' else ('src.jpeg_restoration' if restoration else 'src.jpeg_inference'))
        bundle=ROOT/'artifacts/jpeg_restoration' if restoration else BUNDLE
        arguments=(['--scene-aware','--white-balance'] if scene_aware else []) if color else ['--recipe',recipe] if natural else ['--bundle',str(bundle)]
        try:
            run_worker([sys.executable,'-m',module,str(source),*arguments,
                        '--output',str(result),'--preview-source',str(preview)])
        except ValueError as error:
            detail=str(error)
            if 'camera white balance' in detail:
                detail='This DNG lacks valid camera white balance information and is not supported by the current model.'
            elif 'megapixel limit' in detail:
                detail=('DNG exceeds the 40-megapixel processing limit.' if suffix=='.dng'
                        else 'JPEG exceeds the 64-megapixel processing limit.')
            raise ValueError(detail)
        payload=json.loads(result.read_text())
        payload['timing']['local_worker_seconds']=time.perf_counter()-started
        with np.load(preview,allow_pickle=False) as cached:
            linear=({name:cached[name].copy() for name in ('original','natural_base','adjusted')} if color else
                    {name:cached[name].copy() for name in ('original','adjusted')} if natural else
                    {name:cached[name].copy() for name in ('original','restored')} if restoration
                    else validate_source(cached['linear_rgb'].copy()))
        return payload, linear


def fine_tuning_controls(result,preview):
    base_identity=hashlib.sha256(json.dumps({name:result.get(name) for name in
        ('input_sha256','processing_version','selected_recipe','profile_sha256')},sort_keys=True).encode()).hexdigest()
    with st.expander('Color and local contrast'):
        st.caption('Fine-tune the current recipe. Click Apply to update the result; reset restores this recipe. Overall strength blends all adjustments with the original baseline.')
        with st.form('photo_fine_form'):
            first,second=st.columns(2)
            with first:
                warmth=st.slider('Warmth',-20,20,0,key='photo_warmth',help='Negative is cooler; positive is warmer. Relative color adjustment.')
                tint=st.slider('Tint',-20,20,0,key='photo_tint',help='Negative adds green; positive adds magenta.')
            with second:
                vibrance=st.slider('Vibrance',-20,20,0,key='photo_vibrance',help='Adjusts less saturated colors with skin protection. A small value is usually enough.')
                local=st.slider('Local contrast',0,20,0,key='photo_local_contrast',help='Subtle contrast around lightness transitions. Applies across the image; may emphasize noise and highlight edges.')
            apply=st.form_submit_button('Apply fine tuning',type='primary')
            st.form_submit_button('Reset fine tuning',on_click=reset_fine_tuning)
        if apply:
            st.session_state.pop('photo_export_token',None);st.session_state.pop('photo_saved_png',None)
            settings={'warmth':warmth,'tint':tint,'vibrance':vibrance,'local_contrast':local}
            try:
                if any(settings.values()):
                    with st.spinner('Applying fine tuning at original resolution…'):
                        metadata,pixels=run_fine_tuning(preview['adjusted'],settings)
                    st.session_state['photo_fine_result']=(base_identity,metadata,pixels)
                else:
                    st.session_state.pop('photo_fine_result',None)
            except (ValueError,OSError,subprocess.TimeoutExpired) as error:
                st.error('Fine tuning was not applied: '+str(error))
    tuned=st.session_state.get('photo_fine_result')
    if tuned and tuned[0]==base_identity:
        metadata,pixels=tuned[1:]
        st.caption('Applied fine tuning: '+', '.join(f'{name.replace("_"," ")} {value:+d}' for name,value in metadata['settings'].items()))
        st.caption(f"Fine-tuning job {metadata['local_worker_seconds']:.2f} s · Applied at original resolution")
        if metadata['new_full_channel_fraction']>.001:
            st.warning(f"Fine tuning adds {metadata['new_full_channel_fraction']:.2%} pixels with a full-value channel. Check bright areas and reduce the controls if detail looks lost.")
        return dict(result,fine_tuning=dict(metadata,base_result_identity=base_identity)),{'original':preview['original'],'adjusted':pixels}
    return result,preview


def natural_shadow_controls(result,preview):
    from src.lut_natural import SHADOW_VERSION,apply_shadow_lift
    default=result.get('shadow_adjustment',{})
    if 'natural_base' not in preview or default.get('version')!=SHADOW_VERSION:
        st.info('This result uses an older natural-color version. Process the image again to use Shadow lift.')
        st.stop()
    automatic=round(default['selection']['suggested_strength']*100)
    amount=st.slider('Shadow lift',0,100,automatic,step=5,key='photo_shadow_strength',
                     help='Adjusts darker tones from the fixed color result. Zero removes the extra shadow lift; overall strength zero restores the original image.')/100
    st.caption(f'Automatic shadow suggestion: {automatic}% · Current shadow lift: {amount:.0%}. Color settings stay fixed; dark clouds can also brighten.')
    if amount==default['strength']:
        st.session_state.pop('photo_shadow_result',None)
        return result,preview
    identity=(result['input_sha256'],SHADOW_VERSION,default['input_pixels_sha256'],amount)
    cached=st.session_state.get('photo_shadow_result')
    if not cached or cached[0]!=identity:
        st.session_state.pop('photo_export_token',None);st.session_state.pop('photo_saved_png',None)
        try:
            with st.spinner('Adjusting shadows at original resolution…'):
                metadata,pixels=apply_shadow_lift(preview['natural_base'],amount)
            cached=(identity,metadata,pixels)
            st.session_state['photo_shadow_result']=cached
        except (ValueError,OSError) as error:
            st.session_state.pop('photo_shadow_result',None)
            st.error('Shadow lift was not applied: '+str(error))
            st.stop()
    return dict(result,shadow_adjustment=dict(cached[1],selection=default['selection'])),{'original':preview['original'],'adjusted':cached[2]}


def photo_detail(preview,strength,w,h):
    from src.photo_engine import render_photo
    if st.checkbox('Inspect detail at original size',key='photo_inspect_details'):
        x=st.slider('Detail horizontal position',0,100,50,key='photo_detail_x')
        y=st.slider('Detail vertical position',0,100,50,key='photo_detail_y')
        cw,ch=min(320,w),min(320,h)
        x0,y0=round((w-cw)*x/100),round((h-ch)*y/100)
        crop={name:values[y0:y0+ch,x0:x0+cw] for name,values in preview.items()}
        crop_before,crop_after,_=render_photo(crop,strength)
        st.caption(f'Original-size crop: x={x0}, y={y0}, {cw} × {ch} pixels · No upscaling or added detail')
        original_detail,adjusted_detail=st.columns(2)
        with original_detail: show_png(crop_before,'Original detail',width=cw)
        with adjusted_detail: show_png(crop_after,'Adjusted detail',width=cw)


def show_photo_result(result,preview):
    from src.photo_engine import render_photo
    # Create the visual order first; controls still determine the cached render.
    comparison=st.container(border=True)
    downloads=st.container(border=True)
    with st.expander('Optional adjustments'):
        st.caption('Adjust these only if you prefer a different look.')
        if result['input_status']=='experimental_natural_lut':
            result,preview=natural_shadow_controls(result,preview)
        else:
            result,preview=fine_tuning_controls(result,preview)
        strength=st.slider('Adjustment strength',0,100,100,step=5,key='photo_strength')/100
    baseline,rendered,metadata=render_photo(preview,strength)
    if result['input_status']=='experimental_natural_lut':
        metadata['semantics']=result['natural_color']['natural']['semantics']+' '+result['shadow_adjustment']['semantics']+' Overall strength blends the complete result with the original.'
    w,h=metadata['size']
    with comparison:
        st.subheader('2 · Compare the result')
        st.caption(result.get('input_name','Photo')+f' · {w} × {h}')
        before,after=st.columns(2)
        with before:show_png(baseline,'Original')
        with after:show_png(rendered,'Original retained' if baseline==rendered else 'Adjusted')
        if baseline==rendered:
            st.caption('The current settings retain the original. You can try Optional adjustments below.')
        if strength and result.get('white_balance',{}).get('applied'):
            st.caption('A gentle color-cast correction is included.')
    with downloads:
        png_downloads(rendered,dict(result,export=metadata))
    with st.expander('Inspect photo details'):
        photo_detail(preview,strength,w,h)
        if w*h<1_000_000:
            st.caption('A small source photo keeps its original size; color adjustment cannot recover missing detail.')
        st.caption('Use Optional adjustments to retain a darker mood.')


st.set_page_config(page_title='ShotSense · Automatic photo adjustment', page_icon='📷',
                   layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
    .stMainBlockContainer { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stAppDeployButton"] { display: none; }
    h1 { letter-spacing: -.04em; }
    h2, h3 { letter-spacing: -.02em; }
    [data-testid="stImage"] img { border-radius: 8px; }
    @media (max-width: 640px) {
        .stMainBlockContainer { padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem; }
    }
</style>''',unsafe_allow_html=True)
st.title('ShotSense')
st.write('Natural photo adjustments, made simple.')
st.caption('Upload → Compare → Save · Processed locally on this Mac')
with st.sidebar:
    st.title('Settings')
    st.caption('Automatic works without choosing a mode. Other workflows are available here.')
    advanced=st.expander('Advanced options')
if 'photo_workflow_v2' not in st.session_state:clear_prediction()
with advanced:
    jpeg_workflow = st.radio('Photo workflow', ['Automatic (recommended)', 'Manual exposure', 'Natural adjustment (experimental)', 'Natural color (experimental)', 'Legacy parameter estimates'],
                             key='photo_workflow_v2', on_change=clear_prediction,
                             help='Automatic handles brightness and color for JPEG, and conservative brightness for DNG. Other workflows are optional.')
    scene_aware = st.checkbox('Use scene-aware adjustment (experimental)', key='scene_aware_v1',
                             on_change=clear_prediction,
                             disabled=jpeg_workflow not in ('Automatic (recommended)', 'Natural color (experimental)'),
                             help='JPEG beta: limits changes for night and sunset, and gently corrects color casts when evidence agrees. Personal-photo preference validation is pending.')
automatic=jpeg_workflow=='Automatic (recommended)'
natural=jpeg_workflow=='Natural adjustment (experimental)'
color=jpeg_workflow=='Natural color (experimental)'
recipe='auto'
with advanced:
    if color:
        st.info('Natural color · JPG/JPEG only. Automatically adjusts color and suggests shadow lift. Fine-tune darker tones with Shadow lift; compare before/after. Results are experimental.')
    elif natural:
        choice=st.selectbox('Adjustment recipe', list(PHOTO_RECIPES),
                            key='photo_recipe',on_change=clear_prediction)
        recipe=PHOTO_RECIPES[choice]
        st.info('Experimental · Automatic choice preserves bright and extremely dark images; moderate dimness gets a gentle lift. Personal-photo quality acceptance is pending. Original-resolution PNG export is available in this mode.')
    elif not automatic:
        st.caption('Preview version: ' + VERSION + ' · Original aspect ratio, maximum edge 1600 px')
        st.info('Research prototype · DNG recommendations are absolute values for legacy Camera Raw PV2003. Highlight recovery cannot be mapped directly to modern Highlights. Parameters that have not passed validation are listed separately as experimental outputs.')
        if not (BUNDLE / 'model.json').exists():
            st.warning('The model bundle is unavailable. Restore the verified artifacts/model bundle or follow the training and acceptance instructions in README.md.')
        st.caption('AI low-light enhancement has been withdrawn after real-photo color failures. Manual JPEG processing starts with the original image.')

if not automatic: st.caption('Mode: '+jpeg_workflow)
upload_panel=st.container(border=True)
with upload_panel:
    st.subheader('1 · Add a photo')
    upload = st.file_uploader('Upload a DNG, JPG or JPEG', type=['jpg','jpeg'] if color else ['dng','jpg','jpeg'], on_change=clear_prediction,
                              label_visibility='collapsed',
                              help='Maximum 128 MB; JPEG up to 64 megapixels, DNG up to 40 megapixels. Processed locally; not uploaded to a remote server.')
    st.caption(('JPG · JPEG' if color else 'JPG · JPEG · DNG')+'   /   Your original file is preserved.')
with advanced if automatic else upload_panel:
    left, right = st.columns(2)
    analyze = left.button('Process image', type='primary', disabled=upload is None,
                          help='Automatic processes each new upload. Click here to retry or process it again.')
    sample = right.button('Use project sample',disabled=color,help='The project sample is a DNG; Natural color supports JPG/JPEG.')
attempt=None
if automatic:
    if upload is None:
        if 'automatic_attempt' in st.session_state:clear_prediction()
    else:
        attempt=(upload.name,Path(upload.name).suffix.lower(),hashlib.sha256(upload.getvalue()).hexdigest())
        if scene_aware and Path(upload.name).suffix.lower()!='.dng':
            from src.scene_policy import VERSION as scene_version
            from src.white_balance import VERSION as wb_version
            attempt += (scene_version+':'+wb_version,)
        # ponytail: one attempt per input prevents rerun loops; explicit retry handles transient failures.
        if attempt!=st.session_state.get('automatic_attempt'):analyze=True
pending_recipe=st.session_state.pop('pending_recipe',None)
if pending_recipe is not None:
    sample=pending_recipe[0]
    analyze=not sample
if analyze or sample:
    clear_prediction()
    if automatic and attempt is not None:st.session_state['automatic_attempt']=attempt
    try:
        with st.spinner('Processing the image locally…'):
            if analyze and upload is None:
                raise ValueError('The photo is no longer available. Upload it again.')
            data = upload.getvalue() if analyze else next(iter(sorted((ROOT / 'data/raw/dngs').glob('*.dng')))).read_bytes()
            if pending_recipe is not None and hashlib.sha256(data).hexdigest()!=pending_recipe[1]:
                raise ValueError('The source has changed. Process the current image again.')
            suffix=Path(upload.name).suffix if analyze else '.dng'
            job_color=color or (automatic and suffix.lower()!='.dng')
            job_natural=natural or (automatic and suffix.lower()=='.dng')
            job_scene = bool(scene_aware and job_color)
            st.session_state.prediction = run_job(data,suffix,natural=job_natural,recipe=recipe,color=job_color,scene_aware=job_scene)
            st.session_state.prediction[0]['input_name']=upload.name if analyze else 'Project sample'
            if automatic:
                st.session_state.prediction[0]['automatic_workflow']={'version':'automatic-photo-ui-v1',
                    'backend':('scene-aware-protected-color' if job_scene else 'protected-natural-color') if job_color else 'conservative-native-raw'}
            elif not natural and not color:
                st.session_state.prediction[0]['jpeg_workflow']='manual' if jpeg_workflow.startswith('Manual') else 'legacy'
    except subprocess.TimeoutExpired:
        message='Processing exceeded 60 seconds and was stopped. Please use a smaller image.'
        if automatic:st.session_state['automatic_error']=message
        else:st.error(message)
    except (ValueError, OSError, StopIteration) as error:
        message='Processing failed: ' + str(error)
        if automatic:st.session_state['automatic_error']=message
        else:st.error(message)

if automatic and st.session_state.get('automatic_error'):
    st.error(st.session_state['automatic_error'])
    st.caption('Upload a different photo, or open Advanced options and click Process image to retry.')

if 'prediction' in st.session_state:
    result, preview = st.session_state.prediction
    if result.get('automatic_workflow'):
        show_photo_result(result,preview)
        st.stop()
    if result.get('input_status') == 'experimental_natural_lut':
        show_photo_result(result,preview)
        st.stop()
    if result.get('input_status') == 'experimental_photo_engine':
        from src.photo_engine import render_photo
        comparison=st.container(border=True)
        downloads=st.container(border=True)
        with st.expander('Optional adjustments'):
            strength=st.slider('Adjustment strength',0,100,100,step=5,key='photo_strength')/100
            result,preview=fine_tuning_controls(result,preview)
        baseline,rendered,metadata=render_photo(preview,strength)
        w,h=metadata['size']
        with comparison:
            st.subheader('2 · Compare the result')
            st.caption(result.get('input_name','Photo')+f' · {w} × {h}')
            before,after=st.columns(2)
            with before:
                show_png(baseline,'Before · Neutral RAW development' if result['input_format']=='DNG' else 'Before · Decoded JPEG')
            with after:
                selected=next((label for label,value in PHOTO_RECIPES.items() if value==result['selected_recipe']),result['selected_recipe'])
                tuned=result.get('fine_tuning',{}).get('engine_applied')
                label=('Fine tuning' if result['selected_recipe']=='unchanged' else selected+' + fine tuning') if tuned else selected
                show_png(rendered,'Unchanged · Original baseline preserved' if baseline==rendered else 'Experimental · '+label)
        with downloads:
            png_downloads(rendered,dict(result,export=metadata))
        with st.expander('Recipe choices & processing details'):
            st.caption(f"Full-resolution output: {w} × {h} · Strength {strength:.0%}")
            if result['input_format']=='JPEG' and w*h<1_000_000:
                st.info(f'Small source image: {w} × {h} ({w*h/1_000_000:.2f} MP). PNG preserves these pixels; missing detail and existing JPEG artifacts cannot be recovered by brightness adjustment. The overview does not enlarge the image.')
            processed=next((label for label,value in PHOTO_RECIPES.items() if value==result['requested_recipe']),result['requested_recipe'])
            st.caption('Processed recipe: '+processed)
            is_sample=result.get('input_name')=='Project sample'
            can_retry=is_sample or (upload is not None and upload.name==result.get('input_name'))
            if result['selected_recipe']=='shadows' and result.get('processing_version')=='rawtherapee-recipes-v2':
                st.warning('This cached result uses the older, stronger shadow lift. Reprocess with the revised recipe to retain deeper blacks and limit artifact amplification.')
                st.button('Reprocess with gentler shadows',disabled=not can_retry,on_click=choose_recipe,
                          args=(is_sample,result['input_sha256'],'Lift shadows'),type='primary')
            if baseline==rendered:
                if strength==0:
                    st.info('Strength is 0%: the original baseline is shown. Increase Adjustment strength to apply the processed recipe.')
                elif result['selected_recipe']=='unchanged':
                    if result['selection'].get('reason')=='candidate rejected by clipping guard':
                        fraction=result['selection']['candidate_new_full_channel_fraction']
                        st.info(f'Automatic choice preserved the original: gentle lift introduced {fraction:.2%} new pixels with a full-value channel, above the 0.10% guard. Use Apply gentle lift to explicitly preview that candidate.')
                    else:
                        st.info('Automatic choice preserved the original because it is bright, extremely dark, or has little highlight headroom. Use Apply gentle lift to explicitly try a small adjustment.')
                    st.button('Apply gentle lift',disabled=not can_retry,on_click=choose_recipe,
                              args=(is_sample,result['input_sha256']),type='primary')
                else:
                    st.info('The processed recipe produced unchanged pixels at this strength. Use manual exposure for direct brightness control.')
            if result['requested_recipe']!='shadows':
                st.button('Apply shadow lift',disabled=not can_retry,on_click=choose_recipe,
                          args=(is_sample,result['input_sha256'],'Lift shadows'),type='primary')
            if result['input_format']=='DNG':
                st.caption('Zero strength restores the neutral RAW development. It is not the original RAW sensor data or the parameter-recommendation preview.')
            st.caption('Gentle lift uses +0.35 EV with engine highlight compression. Chroma denoising is optional and may soften detail. This mode cannot reconstruct missing detail or guarantee natural results on every photo.')
            st.caption('Lift shadows preserves the deepest blacks and uses a smoother tonal mask for a moderate lift. Existing noise/compression may remain visible; reduce strength if needed. It does not restore missing detail.')
        with st.expander('Inspect photo details'):
            photo_detail(preview,strength,w,h)
            st.caption(f"Local job {result['timing']['local_worker_seconds']:.2f} s · Engine RawTherapee {result['engine']['version']}")
        st.stop()
    if result.get('input_status') == 'experimental_restoration':
        from src.jpeg_restoration import render_restoration
        st.warning('This AI result has been withdrawn because it can severely alter colors and brightness. Only the preserved input preview is shown. Process the image again to use manual exposure.')
        baseline, _, _ = render_restoration(preview, 0)
        show_png(baseline,'Preserved input preview · AI enhancement not applied')
        local_png_button(baseline,'shotsense-original-preview-'+hashlib.sha256(baseline).hexdigest()[:8]+'.png',
                         'Save original preview to Downloads')
        st.download_button('Download original preview PNG', baseline, file_name='shotsense-original-preview.png', mime='image/png',on_click='ignore')
        st.stop()
    is_jpeg=result.get('input_format')=='JPEG'
    if is_jpeg:
        st.warning('JPEG estimates are experimental: the model was trained on unedited RAW files. Temperature and tint are unavailable; clipped detail cannot be recovered.')
    if isinstance(preview,bytes):
        st.info('The preview has been upgraded. Generate recommendations again to get a high-resolution preview with the original aspect ratio.')
        st.stop()
    st.caption('Current result: ' + result.get('input_name','DNG'))
    st.subheader('2 · Compare the result')
    image_column=st.container()
    parameter_column=st.expander('Parameter estimates & diagnostics')
    with image_column:
        applied={name:value for name,value in result['recommended_absolute'].items()
                 if name in ('Exposure','HighlightRecovery')}
        preview_mode='model_recommendations'
        if is_jpeg:
            conservative=result.get('jpeg_workflow')=='manual'
            preview_mode='unchanged'
            if not conservative and st.checkbox('Apply experimental JPEG estimates to preview',value=False,key='jpeg_preview_estimates'):
                applied={name:value for name,value in result['experimental_absolute'].items()
                         if name in ('Exposure','HighlightRecovery')}
                preview_mode='experimental_estimates'
            if st.checkbox('Adjust JPEG preview manually',value=conservative,key='jpeg_preview_manual'):
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
            show_png(baseline,'Before · Decoded JPEG' if is_jpeg else 'Before · Baseline RAW development')
        with after:
            show_png(rendered,'Unchanged · No visible adjustment' if unchanged else 'Approximate result · Exposure and highlight compression')
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
        png_downloads(rendered,dict(result,preview=preview_metadata),'Download approximate preview PNG','Download parameters JSON')
    with parameter_column:
        st.subheader('Parameter recommendations')
        if result['recommended_absolute']:
            for name, value in result['recommended_absolute'].items():
                st.metric(LABELS[name], f"{value:.2f} {result['units'][name]}")
            st.caption('These parameters outperform the training-median baseline on the fixed validation set. Acceptable error for practical use has not been calibrated.')
        else:
            st.warning('No parameters are validated for JPEG input. Estimates appear below.' if is_jpeg else 'No parameters have passed validation for this model. All outputs are experimental.')
        with st.expander('Experimental outputs',expanded=is_jpeg and result.get('jpeg_workflow')!='manual'):
            for name, value in result['experimental_absolute'].items():
                st.write(f"{LABELS[name]}: {value:.2f} {result['units'][name]}")
            if is_jpeg: st.caption('Temperature and tint: unavailable for JPEG input.')
        st.caption(f"Local job {result['timing']['local_worker_seconds']:.2f} s · Image to estimates {result['timing']['end_to_end_seconds']:.2f} s · Preview source {result['timing'].get('preview_source_seconds',0):.2f} s · Model {result['timing']['model_forward_ms']:.1f} ms")
        st.caption('Model version: ' + result['model_sha256'][:16])
