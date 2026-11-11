"""Score every complete four-image BEV hypothesis with official SALVe weights."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--salve-root',type=Path,required=True)
    p.add_argument('--bev-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.salve_root.resolve()))
    import imageio.v2 as imageio
    import torch
    from salve.models.early_fusion import EarlyFusionCEResnet
    from salve.train_utils import get_val_test_transform
    torch.set_num_threads(4)
    cfg=SimpleNamespace(modalities=['ceiling_rgb_texture','floor_rgb_texture'],resize_h=234,resize_w=234,train_h=224,train_w=224)
    model=torch.nn.DataParallel(EarlyFusionCEResnet(152,False,2,cfg))
    ckpt=args.salve_root/'checkpoints/salve/1200ffbe47d836557d88fef052952337.pth'
    model.load_state_dict(torch.load(ckpt,map_location='cpu')['state_dict'],strict=True);model.eval()
    transform=get_val_test_transform(cfg)
    groups={}
    for path in sorted(args.bev_root.rglob('*.jpg')):
        key=(str(path.parent),path.name.split('_ceiling_rgb_')[0].split('_floor_rgb_')[0])
        groups.setdefault(key,[]).append(path)
    payload={'y_hat':[],'y_true':[],'y_hat_probs':[],'fp0':[],'fp1':[]}
    rows=[];missing=[]
    for key,paths in sorted(groups.items()):
        paths=sorted(paths)
        if len(paths)!=4 or not all('_ceiling_rgb_' in p.name for p in paths[:2]) or not all('_floor_rgb_' in p.name for p in paths[2:]):
            missing.append([str(p) for p in paths]);continue
        tensors=transform(*[imageio.imread(p) for p in paths])
        with torch.inference_mode(): prob=model(*[t.unsqueeze(0) for t in tensors],None,None).softmax(1)[0]
        predicted=int(prob.argmax());label=int('gt_alignment_approx' in paths[0].parts)
        payload['y_hat'].append(predicted);payload['y_true'].append(label)
        payload['y_hat_probs'].append(float(prob[predicted]))
        payload['fp0'].append(str(paths[0].resolve()));payload['fp1'].append(str(paths[1].resolve()))
        rows.append({'files':[str(p) for p in paths],'positive_probability':float(prob[1]),'label_for_evaluation_only':label})
        print(len(rows),key,float(prob[1]),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'batch_0.json').write_text(json.dumps(payload,indent=2))
    (args.output/'audit.json').write_text(json.dumps({'adapted_baseline':True,'scored':len(rows),'incomplete_groups':missing,'rows':rows},indent=2))
    if not rows:raise RuntimeError('No complete BEV groups')


if __name__=='__main__':main()
