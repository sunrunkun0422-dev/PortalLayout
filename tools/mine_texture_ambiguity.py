"""Recompute an exploratory texture shortlist from frozen predictions; no training."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def features(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB').resize((256, 128)), dtype=float) / 255
    result = []
    # Avoid extreme ERP poles. These are bands, NOT semantic surface masks.
    for lo, hi in ((.10, .35), (.65, .90)):
        band = rgb[int(lo*128):int(hi*128)]
        color = np.histogramdd(band.reshape(-1, 3), bins=(8, 8, 8),
                               range=((0, 1),)*3)[0].ravel()
        gray = band.mean(2)
        grad = np.abs(np.diff(gray, axis=1)).ravel()
        texture = np.histogram(grad, bins=16, range=(0, 1))[0].astype(float)
        result.append((color/color.sum(), texture/texture.sum()))
    return result


def similarity(a, b):
    return [float(.5*np.minimum(x[0], y[0]).sum()+.5*np.minimum(x[1], y[1]).sum())
            for x, y in zip(a, b)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root/'experiments/research1_round2_20261024'
    contract = json.loads((run/'run_contract.json').read_text())
    selection = set(contract['house_protocol']['selection'])
    source = run/'R3_final_matching.pairs.jsonl'
    rows = [json.loads(l) for l in source.open()]
    rows = [r for r in rows if r['house_id'] in selection and r['relationship'] == 'cross_room']
    needed = {r['observation_'+s] for r in rows for s in ('a', 'b')}
    paths = {}
    for fp in sorted((root/contract['prediction_cache']/'floors').glob('*.json')):
        f = json.loads(fp.read_text())
        m = root/contract['portallayout.datasets']/'inputs'/f['sample_id']/'manifest.json'
        for obs in json.loads(m.read_text())['observations']:
            if obs['observation_id'] in needed:
                paths[obs['observation_id']] = root.parent/'zind/data'/obs['rgb']['relative_path']
    assert needed == set(paths), 'Missing image mappings'
    feat = {uid: features(path) for uid, path in sorted(paths.items())}
    result = []
    for r in rows:
        sim = similarity(feat[r['observation_a']], feat[r['observation_b']])
        positives = [m for m in r['matches'] if m['target'] == 1]
        negatives = [m for m in r['matches'] if m['target'] == 0]
        group = ('accepted_weak_positive' if positives else
                 'missed_weak_positive' if r['positive_cells'] else 'no_known_positive')
        result.append(dict(pair_id=r['pair_id'], house_id=r['house_id'], floor_uid=r['floor_uid'],
            images=[str(paths[r['observation_'+s]].resolve()) for s in ('a', 'b')],
            upper_similarity=sim[0], lower_similarity=sim[1], similarity=min(sim),
            group=group, positive_cells=r['positive_cells'], matches=r['matches'],
            known_wrong_matches=len(negatives), salve_result=None,
            observation_ids=[r['observation_a'], r['observation_b']]))
    result.sort(key=lambda r: (-r['similarity'], r['pair_id']))
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    (out/'all_pairs.json').write_text(json.dumps(result, indent=2)+'\n')
    shortlist = result[:12]
    for group in ('accepted_weak_positive', 'missed_weak_positive'):
        shortlist += [r for r in result if r['group'] == group][:6]
    shortlist += [r for r in result if r['known_wrong_matches']][:6]
    shortlist = list({r['pair_id']:r for r in shortlist}.values())
    blocks = []
    for r in shortlist:
        canvas = Image.new('RGB', (1000, 540), 'white');draw = ImageDraw.Draw(canvas)
        for i, path in enumerate(r['images']):
            with Image.open(path) as im:
                canvas.paste(im.convert('RGB').resize((1000, 250)), (0, i*270+20))
            draw.text((5, i*270+3), Path(path).parent.parent.name+'/'+Path(path).name, fill='black')
            for a,b in ((.10,.35),(.65,.90)):
                draw.rectangle((0, i*270+20+int(a*250), 999, i*270+20+int(b*250)),outline='orange',width=2)
        filename=r['pair_id']+'.jpg';canvas.save(out/'images'/filename,quality=85)
        blocks.append('<article><h3>'+html.escape(r['house_id']+' / '+r['pair_id'])+'</h3><p>'+
            html.escape(f"upper={r['upper_similarity']:.3f}, lower={r['lower_similarity']:.3f}; "
                        f"{r['group']}; known wrong matches={r['known_wrong_matches']}; SALVe: pending")+
            '</p><img loading="lazy" src="images/'+filename+'"><pre>'+html.escape(json.dumps(r['matches'],indent=2))+'</pre></article>')
    (out/'shortlist.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    audit = json.loads((root/'experiments/salve_dav2_20261029/verifier/audit.json').read_text())
    positives=[r for r in audit['rows'] if r['label_for_evaluation_only']==1]
    summary=dict(scope='development selection houses; frozen R3; exploratory only',
        descriptor='min(upper,lower); each band: 0.5 RGB histogram intersection + 0.5 gray horizontal-gradient histogram intersection',
        surface_masks=False, human_verified=False, pair_count=len(result), image_count=len(paths),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        groups={g:sum(r['group']==g for r in result) for g in sorted({r['group'] for r in result})},
        shortlist_count=len(shortlist), paired_comparison_count=0,
        salve_adapted_positive_hypotheses=len(positives),
        salve_adapted_positive_rejected=sum(r['positive_probability']<.93 for r in positives),
        original_salve_available=False, conclusion='No verified SALVe-fail / ours-pass paired evidence yet')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>纹理相似房间对初筛</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px sans-serif}img{width:100%}article{border-top:2px solid #999;margin-top:30px}pre{white-space:pre-wrap}</style>'
        '<h1>纹理相似房间对：探索性候选</h1><p>橙框为上下区域代理，包含墙壁、家具等，并非真实表面分割。'
        '相似度用于排序，不是概率。先按相似度取前12对，再补充成功、漏检和错误匹配各组最多6对。'
        'accepted_weak_positive 仅表示命中自动弱标签，不代表人工确认、正确位姿或整对无错误。'
        '这些是开发集案例，不是独立测试。SALVe尚未对同一候选评分，不能宣称胜出。</p><pre>'+
        html.escape(json.dumps(summary,indent=2))+'</pre>'+''.join(blocks))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
