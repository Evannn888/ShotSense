"""Optional bounded post-capture WB; presumed neutrals/skin colors are heuristics."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from src.adaptive_lut import validate_rgb

VERSION = 'bounded-deepwb-v2'
PIPELINE_VERSION = 'scene-aware-natural-v4'
BUNDLE = Path(__file__).resolve().parents[1]/'data/external_models/deep_white_balance'
REVISION = 'd70afa227c94152fd8421db75b3ee6ac06a417dc'
MODEL_SHA = 'f237c85fb6fd4f49857a18026dbdb93e57e9871c4e5a505417dfc393c21cc3bb'
LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)
TINT = np.array([1, -(LUMA[0]+LUMA[2])/LUMA[1], 1], dtype=np.float32)


@lru_cache(maxsize=1)
def load_wb_model():
    import onnxruntime as ort
    manifest = json.loads((BUNDLE/'manifest.json').read_text())
    model = BUNDLE/'awb_fp32.onnx'
    if (manifest['revision'] != REVISION or manifest['files'][model.name]['sha256'] != MODEL_SHA
            or hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA):
        raise ValueError('White-balance asset checksum mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])


def wb_preview(rgb):
    image = Image.fromarray(rgb)
    if max(image.size)>16*min(image.size):
        raise ValueError('WB preview skipped for extreme aspect ratio')
    size = tuple(max(1, round(n/max(image.size)*384)) for n in image.size)
    image = image.resize(size, Image.Resampling.BICUBIC)
    padded = tuple(((n+15)//16)*16 for n in size)
    if size != padded:
        image = image.resize(padded, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32)/255


def polynomial(rgb):
    # Author's Hong polynomial, normalized basis; bounded row chunks at native size.
    r,g,b = np.asarray(rgb).reshape(-1,3).T
    return np.column_stack((r,g,b,r*g,r*b,g*b,r*r,g*g,b*b,r*g*b,np.ones_like(r)))


def neutral_evidence(source, predicted):
    y, py = source@LUMA, predicted@LUMA
    chroma, pc = source-y[...,None], predicted-py[...,None]
    mask = (y>=.15)&(y<=.90)&(np.ptp(source,axis=-1)<=.22)
    count = int(mask.sum())
    metadata = {'candidate_count':count,'candidate_fraction':float(mask.mean())}
    if count<128 or mask.mean()<.02:
        return 'insufficient_neutral_candidates',metadata
    before,after = float(np.abs(chroma[mask]).mean()),float(np.abs(pc[mask]).mean())
    direction = (pc-chroma)[mask].mean(axis=0)
    length = float(np.linalg.norm(direction))
    metadata.update(source_candidate_chroma=before, predicted_candidate_chroma=after,
                    mean_chroma_correction=direction.tolist())
    if before<.01 or after>before*.9 or length<.003:
        return 'no_supported_neutral_improvement',metadata
    h,w = mask.shape
    supported = 0;cosines = []
    for ys in (slice(0,h//2),slice(h//2,h)):
        for xs in (slice(0,w//2),slice(w//2,w)):
            part = mask[ys,xs]
            if part.sum()<64:
                continue
            vector = (pc-chroma)[ys,xs][part].mean(axis=0)
            cosine = float(vector@direction/max(float(np.linalg.norm(vector))*length,1e-12))
            cosines.append(cosine);supported += 1
    metadata['quadrant_correction_cosines'] = cosines
    if supported<2 or min(cosines)<.5:
        return 'spatial_neutral_disagreement',metadata
    return None,metadata


def skin_color_weight(a):
    # ponytail: a continuous color proxy also flags non-skin objects and misses skin;
    # use semantic masks only after representative skin/lighting validation.
    cb = (-.168736*a[...,0]-.331264*a[...,1]+.5*a[...,2]+.5)*255
    cr = (.5*a[...,0]-.418688*a[...,1]-.081312*a[...,2]+.5)*255
    return (np.clip((cb-70)/15,0,1)*np.clip((135-cb)/10,0,1)
            *np.clip((cr-125)/15,0,1)*np.clip((185-cr)/15,0,1))


def apply_mapping(rgb, coefficients, tint_only=False):
    validate_rgb(rgb)
    if np.shape(coefficients)!=(11,3) or not np.isfinite(coefficients).all():
        raise ValueError('Invalid WB polynomial')
    output = np.empty_like(rgb)
    black=full=0; luma_max=change_max=0.
    for row in range(0,len(rgb),128):
        source = rgb[row:row+128];a = source.astype(np.float32)/255
        candidate = np.clip((polynomial(a)@coefficients).reshape(a.shape),0,1)
        if not np.isfinite(candidate).all():
            raise ValueError('Nonfinite WB mapping')
        delta = candidate-a
        delta -= (delta@LUMA)[...,None]
        if tint_only:
            # Equal red/blue changes retain the source's warm/cool relationship.
            delta = ((delta@TINT)/(TINT@TINT))[...,None]*TINT
        delta *= .5
        cap = .04-.015*skin_color_weight(a)
        delta *= np.minimum(1,cap/np.maximum(np.abs(delta).max(axis=-1),1e-12))[...,None]
        ceiling = np.where(np.any(source==255,axis=-1),1.,254/255)
        room = np.where(delta>0,(ceiling[...,None]-a)/np.maximum(delta,1e-12),
                        np.where(delta<0,a/np.maximum(-delta,1e-12),1))
        delta *= np.clip(room.min(axis=-1),0,1)[...,None]
        pixels = (np.clip(a+delta,0,1)*255+.5).astype(np.uint8)
        output[row:row+128] = pixels
        change = pixels.astype(np.float32)-source
        luma_max = max(luma_max,float(np.abs(change@LUMA).max()))
        change_max = max(change_max,float(np.abs(change).max()))
        black += int((np.all(pixels==0,axis=-1)&~np.all(source==0,axis=-1)).sum())
        full += int((np.any(pixels==255,axis=-1)&~np.any(source==255,axis=-1)).sum())
    if black or full or luma_max>.501 or change_max>10.501:
        raise ValueError('Bounded WB guard failed')
    return output,{'new_black_pixel_positions':black,'new_full_pixel_positions':full,
                   'maximum_luma_change_codes':luma_max,'maximum_channel_change_codes':change_max,
                   'row_chunk':128,'blend':.5,'channel_cap':.04,'skin_color_proxy_cap':.025,
                   'tint_only':bool(tint_only)}


def warm_evidence(sample, scene):
    y = sample@LUMA
    mask = (y>=.15)&(y<=.90)&(np.ptp(sample,axis=-1)<=.22)
    supported = mask.sum()>=128 and mask.mean()>=.02
    warmth = float((sample[...,0]-sample[...,2])[mask].mean()) if supported else None
    lighting = [v['lighting'] for v in scene['views'].values()]
    sunset = any(v['sunset_sunrise']>=max(v.values())-.04 for v in lighting)
    indoor = all(v['indoor_artificial']>=max(v.values())-.02 for v in lighting)
    protected = bool(supported and warmth>.02 and (sunset or indoor))
    # ponytail: scene/color evidence cannot distinguish deliberate grading from
    # a temperature cast; preserve warmth until independent intent validation.
    return {'protected':protected,'candidate_red_minus_blue':warmth,
            'near_sunset':sunset,'both_near_indoor_artificial':indoor}


def correct_white_balance(rgb, scene, policy):
    validate_rgb(rgb)
    started = time.perf_counter()
    metadata = {'version':VERSION,'status':'skipped','applied':False,'reason':None,
                'source_pixels_sha256':hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest(),
                'semantics':'Bounded learned display-sRGB chroma correction; presumed neutrals and skin colors are heuristics, not illuminant confidence, face masks or camera Kelvin/Tint.'}
    try:
        reason = None
        if scene['status']!='available':
            reason = 'recognition_unavailable'
        elif 'document_preserved' in policy['reasons']:
            reason = 'document_preserved'
        elif policy['source_display_luma_median']<.06:
            reason = 'very_dark_source'
        elif (policy['source_display_luma_median']<.25
              and all(max(v['lighting'],key=v['lighting'].get)=='night'
                      for v in scene['views'].values())):
            reason = 'two_view_dark_night_retained'
        if reason:
            metadata['reason'] = reason
            return metadata,rgb
        sample = wb_preview(rgb)
        # Evaluate the pixels the bounded renderer will actually produce.
        preview = np.rint(sample*255).astype(np.uint8)
        warmth = warm_evidence(sample,scene)
        metadata['warm_evidence'] = warmth
        tensor = np.ascontiguousarray(sample.transpose(2,0,1)[None])
        prediction = load_wb_model().run(None,{'image':tensor})[0]
        if prediction.shape!=tensor.shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid WB model output')
        predicted = np.clip(prediction[0].transpose(1,2,0),0,1)
        coefficients,_,rank,singular = np.linalg.lstsq(polynomial(sample),predicted.reshape(-1,3),rcond=1e-6)
        condition = float(singular[0]/max(singular[-1],1e-12))
        metadata.update(preview_size=[sample.shape[1],sample.shape[0]],rank=int(rank),condition=condition,
                        model_sha256=MODEL_SHA, model_revision=REVISION)
        if rank<11 or condition>1e6:
            metadata['reason'] = 'unsupported_color_mapping'
            return metadata,rgb
        metadata['candidate_attempts'] = []
        for tint_only in ([True] if warmth['protected'] else [False,True]):
            candidate,_ = apply_mapping(preview,coefficients,tint_only=tint_only)
            reason,evidence = neutral_evidence(sample,candidate.astype(np.float32)/255)
            mode = 'tint_only' if tint_only else 'full_chroma'
            metadata['candidate_attempts'].append({'mode':mode,'reason':reason,'evidence':evidence})
            metadata['neutral_evidence'] = evidence
            if reason:
                continue
            adjusted,guards = apply_mapping(rgb,coefficients,tint_only=tint_only)
            metadata.update(status='applied',applied=True,reason='consistent_bounded_neutral_correction',
                            mode=mode,coefficients=coefficients.tolist(),guards=guards,
                            output_pixels_sha256=hashlib.sha256(adjusted).hexdigest())
            return metadata,adjusted
        metadata['reason'] = reason
        return metadata,rgb
    except Exception as error:
        metadata.update(status='fallback',reason='white_balance_unavailable_or_rejected',
                        error=type(error).__name__+': '+str(error)[:200])
        return metadata,rgb
    finally:
        metadata['seconds'] = time.perf_counter()-started
