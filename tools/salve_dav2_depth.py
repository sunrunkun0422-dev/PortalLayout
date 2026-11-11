"""SALVe adaptation: six perspective metric-depth predictions to ERP radial depth.

DAV2 Hypersim predicts optical-axis z, SALVe expects radial range in mm.
No GT geometry is used. Face seams and metric scale bias remain limitations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def sphere(height, width):
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    theta = -(u + .5) / width * 2 * np.pi
    phi = ((v + .5) / height - .5) * np.pi
    return np.stack([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), -np.sin(phi)], -1)


def bases():
    for forward in np.concatenate([np.eye(3), -np.eye(3)]):
        up = np.array([0., 0., 1.]) if abs(forward[2]) < .5 else np.array([0., 1., 0.])
        right = np.cross(forward, up)
        yield forward, right, -np.cross(right, forward)


def render_face(rgb, basis, size):
    f, r, d = basis
    y, x = np.meshgrid((np.arange(size)+.5)*2/size-1, (np.arange(size)+.5)*2/size-1, indexing='ij')
    rays = f + x[..., None]*r + y[..., None]*d
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    h, w = rgb.shape[:2]
    mx = (-np.arctan2(rays[..., 1], rays[..., 0])/(2*np.pi)*w-.5) % w
    my = (np.arcsin(np.clip(-rays[..., 2], -1, 1))/np.pi+.5)*h-.5
    my = np.clip(my, 0, h-1)  # ERP wraps horizontally, never across the poles.
    return cv2.remap(rgb, mx.astype('float32'), my.astype('float32'), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def radial_from_faces(depths, height=512, width=1024):
    rays = sphere(height, width)
    all_bases = list(bases())
    cosines = np.stack([rays @ b[0] for b in all_bases], -1)
    owners = np.argmax(cosines, -1)
    out = np.zeros((height, width), np.float32)
    for i, (f, r, d) in enumerate(all_bases):
        cosine = np.maximum(cosines[..., i], 1e-6)
        s = depths[i].shape[0]
        mx = ((rays @ r)/cosine+1)*s/2-.5
        my = ((rays @ d)/cosine+1)*s/2-.5
        z = cv2.remap(depths[i], mx.astype('float32'), my.astype('float32'), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = owners == i
        out[mask] = (z/cosine)[mask]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('/home/feixia/pythonProject/Depth-Anything-V2'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--floor', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    sys.path.insert(0, str(args.repo/'metric_depth'))
    from depth_anything_v2.dpt import DepthAnythingV2
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = DepthAnythingV2(encoder='vits', features=64, out_channels=[48,96,192,384], max_depth=20)
    net.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    net = net.eval().to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    images = sorted(args.images.glob(args.floor+'_*.jpg'))
    if not images: raise RuntimeError('No panorama images')
    report = {'adapted_baseline': True, 'model':'Depth-Anything-V2-Metric-Hypersim-Small', 'device':device,
              'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'geometry':'6 faces at 90 deg FOV; perspective z / ray cosine -> ERP radial meters',
              'gt_depth_used':False, 'camera_height_assumption_m':1.5, 'panoramas':[]}
    for path in images:
        rgb=cv2.imread(str(path))
        with torch.inference_mode():
            faces=[net.infer_image(render_face(rgb,b,512)).astype('float32') for b in bases()]
        depth=radial_from_faces(faces)
        if not (np.isfinite(depth).all() and (depth>0).all()): raise RuntimeError('Invalid depth')
        out=args.output/(path.stem+'.depth.png')
        assert cv2.imwrite(str(out),np.rint(np.clip(depth,.001,65.535)*1000).astype(np.uint16))
        vis=cv2.applyColorMap(np.uint8(np.clip(depth/20,0,1)*255),cv2.COLORMAP_TURBO)
        cv2.imwrite(str(args.output/(path.stem+'.preview.jpg')),vis)
        report['panoramas'].append({'image':str(path),'depth':str(out),'min_m':float(depth.min()),'median_m':float(np.median(depth)),'max_m':float(depth.max())})
        print(path.name,report['panoramas'][-1]['median_m'],flush=True)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
