"""Present every frozen result with attribution; no fitted decision thresholds."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
LOCAL = ROOT / 'data/user_photo_diagnostics/scene-analysis-public-v1'
TRANSLATE = dict(people='人像', natural_landscape='自然风景', indoor_room='室内空间',
                 urban_architecture='城市建筑', animal_pet='宠物', food='食物',
                 object_still_life='静物', document_graphic='文档或图形',
                 daylight='白天自然光', overcast='阴天', sunset_sunrise='日出／日落',
                 night='夜景', indoor_artificial='室内人工光', indoor_natural='室内自然光',
                 underexposed_daylight='白天欠曝')
NAMES = dict(day_portrait='白天海滩人像', shadow_portrait='蓝橙布光人像', city_night='蓝调城市全景',
             milky_way='银河星空', sunset_silhouette='晚霞与树剪影', kitchen='餐厅室内混合光',
             backlit_landscape='雾中逆光树林', dog='户外幼犬',
             day_portrait_dark_control='白天人像 ×1/8 亮度',
             backlit_landscape_dark_control='逆光树林 ×1/8 亮度')
NOTES = dict(day_portrait='主体合理；阴天与白天接近', shadow_portrait='误判欠曝；不能据此强行提亮',
             city_night='主体合理；欠曝不代表需纠正', milky_way='识别夜景；应保留暗天空',
             sunset_silhouette='识别日落；应保留剪影', kitchen='空间识别合理；光源混合',
             backlit_landscape='取景改变光线判断；需保护光束', dog='主体与自然光判断合理',
             day_portrait_dark_control='未误判夜景；也未识别暗化程度',
             backlit_landscape_dark_control='仍识别风景；未误判夜景')


def sheet(items, rows, faces, path):
    cell_w, cell_h = 440, 474
    cols = 4 if len(items) == 8 else 2
    canvas = Image.new('RGB', (cols * cell_w, ((len(items) + cols - 1) // cols) * cell_h + 44), '#f4f5f6')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 21)
    small = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 15)
    for n, item in enumerate(items):
        key = item['id']; x, y = (n % cols) * cell_w, (n // cols) * cell_h
        with Image.open(item['path']) as source:
            frame = source.convert('RGB')
        pen = ImageDraw.Draw(frame)
        for face in faces.get(key, []):
            bx, by, bw, bh = face['box_bottom_left']
            pen.rectangle((bx * frame.width, (1 - by - bh) * frame.height,
                           (bx + bw) * frame.width, (1 - by) * frame.height), outline='#24cb79', width=2)
        frame = ImageOps.contain(frame, (424, 304), Image.Resampling.LANCZOS)
        canvas.paste(frame, (x + (cell_w - frame.width) // 2, y + (312 - frame.height) // 2))
        row = rows[(key, 'full_frame_letterbox')]
        subject = row['groups']['subject']['top_two'][0]['label']
        light = row['groups']['lighting']['top_two'][0]['label']
        draw.text((x + 10, y + 318), f'{n+1}. {NAMES[key]}', font=font, fill='#111111')
        draw.text((x + 10, y + 351), f'主体：{TRANSLATE[subject]}   光线：{TRANSLATE[light]}', font=font, fill='#183d52')
        draw.text((x + 10, y + 385), NOTES[key], font=font, fill='#6d3630')
        parent = next((i for i in ORIGINALS if i['id'] == item.get('parent_id')), item)
        draw.text((x + 10, y + 420), parent['author'][:48], font=small, fill='#555555')
        draw.text((x + 10, y + 443), parent['license'] + ' · 缩小／排列／标注；详见报告来源', font=small, fill='#555555')
    draw.text((12, canvas.height - 32), '全画面模型结果；绿色框为原生人脸检测。仅验证识别，未对照片调色。联系图 CC BY-SA 4.0。', font=small, fill='#333333')
    canvas.save(path)


if __name__ == '__main__':
    inputs = json.loads((FOLDER / 'inputs.json').read_text())
    expected = json.loads((FOLDER / 'expected_labels.json').read_text())
    result = json.loads((FOLDER / 'results.json').read_text())
    native = json.loads((FOLDER / 'vision_results.json').read_text())
    rows = {(r['id'], r['view']): r for r in result['rows'] if r['run'] == 0}
    faces = {r['id']: r['faces'] for r in native['rows'] if r['pass'] == 0}
    ORIGINALS = [i for i in inputs if not i.get('synthetic')]
    summary = []
    for item in inputs:
        key = item['id']; crop, full = [rows[(key, v)] for v in ('stock_center_crop', 'full_frame_letterbox')]
        subject = full['groups']['subject']['top_two'][0]['label']
        light = full['groups']['lighting']['top_two'][0]['label']
        with Image.open(item['path']) as im:
            srgb = np.asarray(im.convert('RGB'), dtype=np.float64) / 255
        linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
        luminance = linear @ np.array([.2126, .7152, .0722])
        summary.append(dict(id=key, name=NAMES[key], synthetic=item.get('synthetic', False),
                full_subject=subject, crop_subject=crop['groups']['subject']['top_two'][0]['label'],
                subject_matches_declared=subject in expected[key]['subject_acceptable'],
                full_lighting=light, crop_lighting=crop['groups']['lighting']['top_two'][0]['label'],
                lighting_top_in_declared_set=light in expected[key]['lighting_acceptable'],
                lighting_ambiguous=expected[key].get('lighting_ambiguous', False),
                subject_top_two_gap=full['groups']['subject']['top_two_gap'],
                lighting_top_two_gap=full['groups']['lighting']['top_two_gap'],
                linear_luminance_q05_q50_q95=[float(v) for v in np.quantile(luminance, [.05, .5, .95])],
                native_faces=len(faces[key]), analysis=NOTES[key]))
    timings = [r['model_ms'] for r in result['rows'] if r['run'] == 1]
    (FOLDER / 'summary.json').write_text(json.dumps(dict(items=summary,
        original_count=len(ORIGINALS), synthetic_count=len(inputs) - len(ORIGINALS),
        original_lighting_view_flips=[i['id'] for i in summary if not i['synthetic'] and i['full_lighting'] != i['crop_lighting']],
        warm_model_p50_p95_ms=[float(v) for v in np.percentile(timings, [50, 95])]),
        ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    sheet(ORIGINALS, rows, faces, LOCAL / 'recognition_results.png')
    controls = []
    for item in inputs:
        if item.get('synthetic'):
            controls.extend([next(i for i in ORIGINALS if i['id'] == item['parent_id']), item])
    sheet(controls, rows, faces, LOCAL / 'darkening_results.png')
    print(json.dumps({'originals': len(ORIGINALS), 'controls': len(controls) // 2,
                      'subject_matches': sum(i['subject_matches_declared'] for i in summary if not i['synthetic'])}))
