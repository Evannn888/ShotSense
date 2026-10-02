"""Local DNG parameter recommendations; each RAW job runs in a bounded process."""
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
LABELS = {'Exposure':'曝光', 'Contrast':'对比度', 'Saturation':'饱和度',
          'Temperature':'色温', 'Tint':'色调', 'HighlightRecovery':'高光恢复（旧流程）'}


@st.cache_resource
def job_lock():
    # ponytail: one RAW job at a time; use a queue if remote multi-user support is needed.
    return threading.Lock()


def clear_prediction():
    st.session_state.pop('prediction', None)


def run_job(data):
    started=time.perf_counter()
    if not 0 < len(data) <= 128 * 1024 * 1024:
        raise ValueError('文件必须小于 128 MB。')
    with job_lock(), tempfile.TemporaryDirectory(prefix='shotsense-') as directory:
        folder = Path(directory)
        source, result, preview = [folder / name for name in ('input.dng','result.json','preview-source.npz')]
        source.write_bytes(data)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        process = subprocess.run([sys.executable, '-m', 'src.inference', str(source),
                                  '--bundle', str(BUNDLE), '--output', str(result), '--preview-source', str(preview)],
                                 cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        if process.returncode:
            detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else '无法解码 DNG'
            if 'camera white balance' in detail:
                detail='此 DNG 缺少有效的相机白平衡信息，当前模型不支持。'
            elif '40 megapixel' in detail:
                detail='图像超过 4000 万像素的处理上限。'
            raise ValueError(detail)
        payload=json.loads(result.read_text())
        payload['timing']['local_worker_seconds']=time.perf_counter()-started
        with np.load(preview,allow_pickle=False) as cached:
            linear=validate_source(cached['linear_rgb'].copy())
        return payload, linear


st.set_page_config(page_title='ShotSense · 调色参数建议', page_icon='📷', layout='wide')
st.title('ShotSense')
st.caption('预览版本：' + VERSION + ' · RAW 等比高清预览')
st.write('从未调色 DNG 提取图像特征，推荐曝光与色彩参数。')
st.info('研究原型 · 建议为绝对值，使用旧版 Camera Raw PV2003 参数。高光恢复不能直接映射为现代 Highlights；未通过验证的参数会单列为实验输出。')

if not (BUNDLE / 'model.json').exists():
    st.warning('模型尚未准备好。请先完成全量训练和 ONNX 导出。')
    st.code('venv/bin/python -m src.train\nvenv/bin/python -m src.export_onnx')
    st.stop()

upload = st.file_uploader('上传未调色 DNG', type=['dng'], on_change=clear_prediction,
                          help='最大 128 MB、4000 万像素；本地处理，不上传远程服务器。')
left, right = st.columns(2)
analyze = left.button('生成参数建议', type='primary', disabled=upload is None)
sample = right.button('使用项目样例')
if analyze or sample:
    clear_prediction()
    try:
        with st.spinner('正在读取 RAW、提取特征并推理…'):
            data = upload.getvalue() if analyze else next(iter(sorted((ROOT / 'data/raw/dngs').glob('*.dng')))).read_bytes()
            st.session_state.prediction = run_job(data)
            st.session_state.prediction[0]['input_name']=upload.name if analyze else '项目样例'
    except subprocess.TimeoutExpired:
        st.error('处理超过 60 秒，任务已终止。请使用较小的 DNG。')
    except (ValueError, OSError, StopIteration) as error:
        st.error('处理失败：' + str(error))

if 'prediction' in st.session_state:
    result, preview = st.session_state.prediction
    if isinstance(preview,bytes):
        st.info('预览已升级，请重新生成建议以获得保持比例的高清效果。')
        st.stop()
    st.caption('当前结果：' + result.get('input_name','DNG'))
    image_column, parameter_column = st.columns([3, 2])
    with image_column:
        applied={name:value for name,value in result['recommended_absolute'].items()
                 if name in ('Exposure','HighlightRecovery')}
        strength=st.slider('建议应用强度',0,100,100,step=5,key='preview_strength')/100
        protect=st.checkbox('保护高光',value=True,key='preview_protect')
        baseline,rendered,preview_metadata=render_linear_preview(preview,applied,strength,protect)
        before,after=st.columns(2)
        with before:
            st.image(baseline, caption='应用前 · RAW 基准显影', width='stretch')
        with after:
            st.image(rendered, caption='近似应用后 · 曝光与高光压缩', width='stretch')
        st.caption(f"近似预览（{preview_metadata['size'][0]} × {preview_metadata['size'][1]}）：保持原比例，仅模拟曝光与高光压缩，实验参数未应用。效果不等同于 Lightroom，不保证恢复原始剪裁细节。")
        effective=preview_metadata['applied_parameters']
        st.caption(f"实际预览：曝光 {effective['Exposure']:.2f} EV · 高光压缩代理 {effective['HighlightRecovery']:.2f} · 新增通道满值像素 {preview_metadata['new_full_channel_fraction']:.2%}")
        st.caption(f"基准预览达到上限：至少一个通道 {preview_metadata['source_channel_ceiling_fraction']:.2%} · 三通道全白 {preview_metadata['source_white_ceiling_fraction']:.2%}")
        if preview_metadata['source_channel_ceiling_fraction'] >= .001:
            st.warning('基准预览已有通道达到上限，可能来自 RAW 显影或显示色域裁剪。压暗这些区域不代表恢复纹理；全白区域可能仍显得平坦。')
        if not protect:
            st.warning('高光保护已关闭；直接曝光可能造成新增剪裁。')
        st.download_button('下载近似效果 PNG', rendered, file_name='shotsense-approximate-preview.png',mime='image/png')
    with parameter_column:
        st.subheader('参数建议')
        if result['recommended_absolute']:
            for name, value in result['recommended_absolute'].items():
                st.metric(LABELS[name], f"{value:.2f} {result['units'][name]}")
            st.caption('这些参数在固定验证集上优于训练集中位数基线，业务可接受误差尚未标定。')
        else:
            st.warning('当前模型没有通过验证的参数，所有输出仅供实验。')
        with st.expander('实验输出'):
            for name, value in result['experimental_absolute'].items():
                st.write(f"{LABELS[name]}：{value:.2f} {result['units'][name]}")
        export_payload=dict(result,preview=preview_metadata)
        st.download_button('下载参数 JSON', json.dumps(export_payload, ensure_ascii=False, indent=2),
                           file_name='shotsense-parameters.json', mime='application/json')
        st.caption(f"本地任务 {result['timing']['local_worker_seconds']:.2f} 秒 · RAW 到建议 {result['timing']['end_to_end_seconds']:.2f} 秒 · 预览源 {result['timing'].get('preview_source_seconds',0):.2f} 秒 · 模型 {result['timing']['model_forward_ms']:.1f} 毫秒")
        st.caption('模型版本：' + result['model_sha256'][:16])
