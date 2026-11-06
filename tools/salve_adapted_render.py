"""Render a fixed floor using precomputed adapter depths, never HoHoNet fallback."""
import argparse
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--salve-root',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--building',default='0007');p.add_argument('--floor',default='floor_02')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();sys.path.insert(0,str(args.salve_root.resolve()))
    import cv2
    import numpy as np
    for image in (args.raw_root/args.building/'panos').glob(args.floor+'_*.jpg'):
        depth=args.run/'depth'/args.building/(image.stem+'.depth.png')
        array=cv2.imread(str(depth),cv2.IMREAD_UNCHANGED)
        if array is None or array.shape!=(512,1024) or array.dtype!=np.uint16 or np.any(array==0):
            raise RuntimeError(f'Missing or invalid adapter depth: {depth}')
    from scripts.render_dataset_bev import render_building_floor_pairs
    b=args.run.resolve()
    render_building_floor_pairs(str(b/'depth'),str(b/'bev'),str(b/'hypotheses'),str(args.raw_root.resolve()),args.building,args.floor,None,['rgb_texture'],True,args.workers)


if __name__=='__main__':main()
