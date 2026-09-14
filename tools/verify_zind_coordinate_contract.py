#!/usr/bin/env python3
"""Numerically compare the new projection/transform with official ZInD code."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset.zind_observable_pairs import projection
from dataset.zind_pair_mining import transform_points,zind_transform_matrix
from tools.build_zind_observable_pairs import sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--zind',type=Path,default=ROOT.parent/'zind');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    train=json.loads((a.zind/'zind_partition.json').read_text())['train'];items=[]
    for house in sorted(train)[:8]:
        data=json.loads((a.zind/'data'/house/'zind_data.json').read_text())
        for fid,rooms in data['merger'].items():
            for cr,partials in rooms.items():
                for pr,panos in partials.items():
                    for pid,r in panos.items():
                        if r.get('layout_raw',{}).get('vertices'):
                            items.append({'ref':f'{house}/{fid}/{cr}/{pr}/{pid}', 'transform':r['floor_plan_transformation'],'points':r['layout_raw']['vertices']})
    code="""import json,sys,numpy as np
# Official utils.py has an obsolete type annotation; arithmetic is unchanged.
np.float=float
from transformations import Transformation2D,TransformationSpherical
x=json.load(sys.stdin);out=[]
for item in x:
 p=np.array(item['points']);z=np.column_stack((p,np.full(len(p),.7)))
 out.append({'global':Transformation2D.from_zind_data(item['transform']).to_global(p).tolist(),'pixel':TransformationSpherical.cartesian_to_pixel(z,1024).tolist()})
print(json.dumps(out))"""
    official=json.loads(subprocess.run([sys.executable,'-B','-c',code],cwd=str(a.zind/'code'),input=json.dumps(items),capture_output=True,text=True,check=True).stdout)
    xyerr=[];pixelerr=[]
    for item,ref in zip(items,official):
        q=transform_points(item['points'],zind_transform_matrix(item['transform']));xyerr.append(float(np.max(np.abs(q-ref['global']))))
        for point,pixel in zip(item['points'],ref['pixel']):
            proj=projection([point,point],[.7,.7]);u=proj['endpoint_u'][0];v=proj['top_v'][0]
            du=abs((u-pixel[0]/1023+.5)%1-.5)*1023;dv=abs(v*511-pixel[1]);pixelerr.append(max(du,dv))
    assert max(xyerr)<1e-10 and max(pixelerr)<1e-9
    report={'status':'passed','train_houses':sorted(train)[:8],'panorama_transforms_compared':len(items),
        'max_floor_xy_absolute_error':max(xyerr),'max_projection_pixel_error':max(pixelerr),
        'projection_resolution':[1024,512],'longitude_comparison':'circular; u=0 and u=1 represent the same seam',
        'official_files_sha256':{str(f.relative_to(a.zind)):sha(f) for f in [a.zind/'code/transformations.py',a.zind/'code/render.py',a.zind/'README.md',a.zind/'data_organization.md']},
        'compatibility_shim':'subprocess-only np.float=float for official legacy type annotation; official files unmodified',
        'test_annotations_read':False}
    write(a.output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
