"""Restrained source-adaptive color with monotone tone and dark-detail protection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.adaptive_lut import adjust_rgb, validate_rgb

VERSION = 'natural-adaptive-lut-v2'
SHADOW_VERSION = 'natural-shadow-curve-v1'
LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)


def smoothstep(low, high, values):
    x = np.clip((values-low)/(high-low), 0, 1)
    return x*x*(3-2*x)


def natural_settings(original, adjusted):
    validate_rgb(original); validate_rgb(adjusted)
    if original.shape != adjusted.shape: raise ValueError('Natural LUT dimensions differ')
    h,w = original.shape[:2]
    a = original[::max(1,(h+255)//256),::max(1,(w+255)//256)].astype(np.float32)/255
    b = adjusted[::max(1,(h+255)//256),::max(1,(w+255)//256)].astype(np.float32)/255
    ay,by = a@LUMA,b@LUMA
    delta = (b-by[...,None])-(a-ay[...,None])
    change = float(np.percentile(np.max(np.abs(delta),axis=-1),95))
    shadow = float(np.percentile(ay,30))
    # ponytail: sampled scene statistics cannot identify artistic intent; keep this opt-in.
    return {'color_strength': min(.35,.04/max(change,.04)),
            'tone_lift': .15*float(np.clip((.35-shadow)/.25,0,1)),
            'sample_color_change_p95': change, 'sample_luminance_q30': shadow,
            'sample_size': [a.shape[1],a.shape[0]]}


def protect_rgb(original, adjusted, caps=None):
    """Keep pretrained color changes bounded; use an independent non-darkening tone curve."""
    started = time.perf_counter(); settings = natural_settings(original,adjusted)
    if caps is not None:
        for name, ceiling in (('color_strength', .35), ('tone_lift', .15)):
            value = caps[name]
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 <= value <= ceiling:
                raise ValueError('Invalid scene adjustment cap')
            settings[name] = min(settings[name], float(value))
    output = np.empty_like(original)
    new_black = new_full = 0; minimum_shadow_delta = 0.
    for row in range(0,len(original),128):
        source = original[row:row+128]; a = source.astype(np.float32)/255
        b = adjusted[row:row+128].astype(np.float32)/255
        y,by = a@LUMA,b@LUMA
        color = a-y[...,None]
        mask = (smoothstep(.03,.25,y)*(1-smoothstep(.75,.98,y))
                *smoothstep(0,.08,np.ptp(a,axis=-1)))
        change = ((b-by[...,None])-color)*(settings['color_strength']*mask[...,None])
        change *= np.minimum(1,.05/np.maximum(np.max(np.abs(change),axis=-1),1e-7))[...,None]
        color += change
        headroom = np.where(np.any(source==255,axis=-1),1.,254/255).astype(np.float32)
        target_y = np.minimum(y+settings['tone_lift']*y*(1-y)**2,headroom)
        room = np.where(color>0,(headroom-target_y)[...,None]/np.maximum(color,1e-7),
                        np.where(color<0,target_y[...,None]/np.maximum(-color,1e-7),1))
        scale = np.clip(np.min(room,axis=-1),0,1)
        pixels = ((np.clip(target_y[...,None]+color*scale[...,None],0,1)*255)+.5).astype(np.uint8)
        output[row:row+128] = pixels
        new_black += int((np.all(pixels==0,axis=-1)&~np.all(source==0,axis=-1)).sum())
        new_full += int((np.any(pixels==255,axis=-1)&~np.any(source==255,axis=-1)).sum())
        shadow = y<=.35
        if np.any(shadow):
            minimum_shadow_delta = min(minimum_shadow_delta,float(((pixels.astype(np.float32)-source)@LUMA)[shadow].min()))
    count = original.shape[0]*original.shape[1]
    if new_black or new_full or minimum_shadow_delta < -.501:
        raise ValueError('Natural LUT protection failed')
    metadata = {'version':VERSION,'settings':settings,'size':[original.shape[1],original.shape[0]],
        'new_black_pixel_fraction':new_black/count,'new_full_channel_fraction':new_full/count,
        'minimum_shadow_luminance_delta_codes':minimum_shadow_delta,'row_chunk':128,
        'input_pixels_sha256':hashlib.sha256(np.ascontiguousarray(original)).hexdigest(),
        'output_pixels_sha256':hashlib.sha256(np.ascontiguousarray(output)).hexdigest(),
        'seconds':time.perf_counter()-started,'shadow_protection':True,'neutral_color_protection':True,
        'semantics':'Bounded global display-sRGB color change with independent monotone mild brightness; not semantic masks, camera Kelvin/EV, denoising, sharpening or reconstructed detail.'}
    return metadata,output


def adjust_natural_rgb(rgb, model=None, caps=None):
    started = time.perf_counter()
    pretrained,raw = adjust_rgb(rgb,model)
    protection,adjusted = protect_rgb(rgb,raw,caps)
    return {'pretrained':pretrained,'natural':protection,'seconds':time.perf_counter()-started},adjusted


def shadow_settings(original):
    validate_rgb(original)
    h,w=original.shape[:2]
    sample=original[::max(1,(h+255)//256),::max(1,(w+255)//256)].astype(np.float32)/255
    y=sample@LUMA; q30=float(np.percentile(y,30))
    fraction=float(((y>=.02)&(y<.35)).mean())
    amount=round(75*float(np.clip((.4-q30)/.3,0,1))*min(1,fraction/.25)/5)*5
    return {'suggested_strength':amount/100,'sample_luminance_q30':q30,
            'sample_shadow_fraction':fraction,'sample_size':[sample.shape[1],sample.shape[0]],
            'semantics':'Provisional brightness-distribution rule; photographic intent is unknown.'}


def shadow_curve(y,strength):
    if (not isinstance(strength,(int,float,np.integer,np.floating)) or isinstance(strength,(bool,np.bool_))
            or not np.isfinite(strength) or not 0<=strength<=1):
        raise ValueError('Shadow strength must be between0and1')
    return y+strength*1.8*y*(1-y)**6*smoothstep(.008,.045,y)*(1-smoothstep(.25,.55,y))


def apply_shadow_lift(rgb,strength):
    """Brightness-only common gain from the fixed natural base; never accumulate edits."""
    validate_rgb(rgb);shadow_curve(np.array([0.],dtype=np.float32),strength)
    started=time.perf_counter();output=rgb.copy();new_full=new_black=0
    for row in range(0,len(rgb),128) if strength else ():
        source=rgb[row:row+128];pixels=source.astype(np.float32)/255
        y=pixels@LUMA;target=shadow_curve(y,strength)
        ceiling=np.where(np.any(source==255,axis=-1),1.,254/255).astype(np.float32)
        gain=np.minimum(target/np.maximum(y,1e-7),ceiling/np.maximum(pixels.max(axis=-1),1e-7))
        adjusted=(np.clip(pixels*gain[...,None],0,1)*255+.5).astype(np.uint8)
        output[row:row+128]=adjusted
        new_black+=int((np.all(adjusted==0,axis=-1)&~np.all(source==0,axis=-1)).sum())
        new_full+=int((np.any(adjusted==255,axis=-1)&~np.any(source==255,axis=-1)).sum())
        if np.any(adjusted<source):raise ValueError('Shadow lift unexpectedly darkened pixels')
    count=rgb.shape[0]*rgb.shape[1]
    if new_black or new_full:raise ValueError('Shadow lift protection failed')
    return {'version':SHADOW_VERSION,'strength':float(strength),'size':[rgb.shape[1],rgb.shape[0]],
        'new_black_pixel_fraction':new_black/count,'new_full_channel_fraction':new_full/count,
        'input_pixels_sha256':hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest(),
        'output_pixels_sha256':hashlib.sha256(np.ascontiguousarray(output)).hexdigest(),
        'seconds':time.perf_counter()-started,'row_chunk':128,
        'semantics':'Brightness-only common display-sRGB RGB gain; deep-black and upper-midtone taper, bounded channel headroom. No semantic region detection or spatial processing.'},output


def process_jpeg(path, scene_aware=False, white_balance=False):
    from src.jpeg_inference import decode_jpeg
    if white_balance and not scene_aware:
        raise ValueError('Bounded white balance requires scene-aware protection')
    started = time.perf_counter(); path = Path(path)
    original,profile = decode_jpeg(path)
    scene = policy = None
    if scene_aware:
        from src.scene_policy import analyze_scene, scene_limits
        scene = analyze_scene(original)
        policy = scene_limits(original, scene)
    working = original
    wb = None
    if white_balance:
        from src.white_balance import correct_white_balance
        wb, working = correct_white_balance(original, scene, policy)
    metadata,base = adjust_natural_rgb(working, caps=policy['limits'] if policy else None)
    selection=shadow_settings(original)
    if policy:
        selection['unconstrained_suggested_strength'] = selection['suggested_strength']
        selection['scene_limit'] = policy['limits']['shadow_strength']
        selection['suggested_strength'] = min(selection['suggested_strength'], selection['scene_limit'])
    shadows,adjusted=apply_shadow_lift(base,selection['suggested_strength'])
    if policy:
        malformed = (not isinstance(adjusted, np.ndarray) or adjusted.dtype != np.uint8
                     or adjusted.shape != original.shape)
        black = full = 0
        for row in range(0, len(original), 128) if not malformed else ():
            a, b = original[row:row+128], adjusted[row:row+128]
            black += int((np.all(b==0, axis=-1) & ~np.all(a==0, axis=-1)).sum())
            full += int((np.any(b==255, axis=-1) & ~np.any(a==255, axis=-1)).sum())
        policy['final_check'] = {'malformed': malformed, 'new_black_pixel_positions': black, 'new_full_pixel_positions': full}
        if malformed or black or full:
            policy['status'] = 'candidate_rejected'
            policy['reasons'].append('final_endpoint_guard')
            metadata['natural']['rejected_output_pixels_sha256'] = metadata['natural']['output_pixels_sha256']
            base = original.copy()
            metadata['natural']['output_pixels_sha256'] = hashlib.sha256(base).hexdigest()
            metadata['natural']['applied'] = False
            if wb:
                wb.update(applied=False,status='rolled_back',reason='final_endpoint_guard')
            selection['suggested_strength'] = 0.
            shadows, adjusted = apply_shadow_lift(base, 0.)
    result = {'schema_version':1,'input_format':'JPEG','input_status':'experimental_natural_lut',
        'processing_version':VERSION,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'selected_recipe':'natural-color','natural_color':metadata,'output_size':metadata['natural']['size'],
        'shadow_adjustment':dict(shadows,selection=selection),
        'jpeg_color':{'embedded_icc_converted_to_srgb':profile,'without_icc':'assumed sRGB'},
        'quality_status':'experimental; independent personal-photo preference acceptance pending',
        'timing':{'end_to_end_seconds':time.perf_counter()-started}}
    if policy:
        result.update(processing_version=policy['version'], scene_analysis=scene, scene_policy=policy)
    if wb:
        from src.white_balance import PIPELINE_VERSION
        policy['white_balance'] = {key:wb[key] for key in ('version','status','reason')}
        result.update(processing_version=PIPELINE_VERSION,white_balance=wb)
    return result,{'original':original,'natural_base':base,'adjusted':adjusted}


if __name__=='__main__':
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jpeg',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preview-source',type=Path,required=True)
    parser.add_argument('--scene-aware', action='store_true', help='Optional local scene-restraint research beta')
    parser.add_argument('--white-balance', action='store_true', help='Optional bounded AWB; requires --scene-aware')
    args = parser.parse_args(); torch.set_num_threads(1)
    result,pixels = process_jpeg(args.jpeg, scene_aware=args.scene_aware, white_balance=args.white_balance)
    np.savez_compressed(args.preview_source,**pixels)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
