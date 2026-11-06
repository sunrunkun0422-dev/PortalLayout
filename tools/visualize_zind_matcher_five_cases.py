"""Render five qualitative T3 val cases from frozen artifacts; no model loading."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bi_layout_five_cases_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})

from tools.visual_audit_zind_candidate_failures import interval_pixel_spans, interval_center

F = ROOT/'outputs/zind_geometry_val_full/t3_frozen_m2'
A = ROOT/'experiments/zind_e2e_v3/g0_seed123/T5_candidate_recall/audit'
B = ROOT/'experiments/zind_e2e_v3/g0_seed123/T5_bev_verifier/preflight'
V = A/'visual_audit'
NAMES = ['correct_match', 'dustbin_reject', 'wrong_selection',
         'negative_correct_reject', 'negative_false_accept']
TITLES = ['Correct Match', 'GT Recalled → Rejected by Dustbin',
          'GT Recalled → Wrong Opening Selected',
          'Non-adjacent Pair → Correctly Rejected',
          'Non-adjacent Pair → False Positive Match']
COLORS = ['#16754c', '#986214', '#b43c32', '#16754c', '#b43c32']
GT, CAND, SELECTED, RAW = '#198347', '#007fad', '#d24f25', '#7653a4'
FOOTNOTE = 'GT is used for evaluation only and is not model input.'
HASHES = {}


def read(path, lines=False):
    path = Path(path).resolve()
    if any('test' in p.lower() for p in path.parts):
        raise ValueError(f'test path forbidden: {path}')
    blob = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(blob).hexdigest()
    if path.suffix == '.gz':
        blob = gzip.decompress(blob)
    return [json.loads(l) for l in blob.splitlines() if l] if lines else json.loads(blob)


def category(row):
    p, e, g = row['prediction'], row['portallayout.evaluation'], row['ground_truth']
    if not g['is_match']:
        return 4 if p['is_match'] else 3
    if not e['gt_candidate_recalled']:
        return None
    if not p['is_match']:
        return 1
    return 0 if e['selected_candidate_correct'] else 2


def label(pair):
    return f'A{pair[0]} ↔ B{pair[1]}'


def load_case(row, record, frozen_map, score_map, audit_rows, data_root):
    pid = row['pair_id']
    f = read(frozen_map[pid]); s = read(score_map[pid])
    assert f['pair_id'] == s['pair_id'] == record['id'] == pid
    assert s['split'] == 'val' and s['test_data_accessed'] is False
    assert f['test_data_accessed'] is False and f['gt_available_to_prediction'] is False
    na, nb = f['candidate_count_A'], f['candidate_count_B']
    assert (na, nb) == (s['candidate_count_A'], s['candidate_count_B'])
    scores = np.array(f['candidate_pair_scores'])[:na, :nb]
    rescored = np.array(s['candidate_pair_scores'])[:na, :nb]
    logits = np.array(s['candidate_assignment_logits'])[:na, :nb]
    assert np.isfinite(scores).all() and np.isfinite(logits).all()
    order = sorted([(i, j) for i in range(na) for j in range(nb)], key=lambda ij: (-scores[ij], ij))
    raw = list(order[0]); final = list(f['best_candidate_pair'])
    assert raw == list(np.unravel_index(rescored.argmax(), rescored.shape))
    assert final == s['best_candidate_pair'] == row['prediction']['best_candidate_pair']
    assert bool(f['is_match']) == bool(row['prediction']['is_match']) == (final != [-1, -1])
    # Dustbin evidence is a directly stored shared scalar, not probability inversion.
    direct = [float(x) for k in ['row_dustbin_logits', 'column_dustbin_logits']
              for x in s.get(k, []) if np.isfinite(x) and x > -1e20]
    evidence = read(V/'visualization_manifest.json')['dustbin_recovery_evidence']
    db = evidence['shared_dustbin_logit']
    assert all(abs(x-db) < 1e-7 for x in direct)
    accepted_cells = [ij for ij in order if logits[ij] > db]
    assert (list(accepted_cells[0]) if accepted_cells else [-1, -1]) == final
    intervals = [f['predicted_intervals_A'], f['predicted_intervals_B']]
    assert intervals == [s['intervals_A'], s['intervals_B']]
    positive = bool(row['ground_truth']['is_match'])
    assert bool(record['supervision']['is_match']) == positive
    targets = [list(x) for x in row['portallayout.evaluation']['target_predicted_candidate_pairs']] if positive else []
    gt_intervals = []
    if positive:
        ar = audit_rows[pid]
        assert sorted(targets) == sorted(ar['legacy']['target_pairs'])
        for side, upper in [('a', 'A'), ('b', 'B')]:
            index = record['supervision']['target_candidate_'+side]
            interval = record['candidates_'+side][index]['tokenInterval']
            assert interval == ar['target_'+upper]['target_interval']
            gt_intervals.append(interval)
    rank = lambda ij: order.index(tuple(ij)) + 1
    if positive:
        assert targets
        gt_pair = min(targets, key=rank)
    else:
        gt_pair = None
    images = []
    image_paths = []
    for side in ['a', 'b']:
        p = data_root/record['image_'+side]
        assert p.is_file()
        if 'test' in p.parts:
            raise ValueError('test RGB forbidden')
        HASHES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            im.load(); images.append(im.convert('RGB').copy())
        image_paths.append(str(p))
    kind = category(row)
    # Additional semantic gates for the particular illustrations requested.
    if kind == 0:
        assert raw == final and final in targets
    if kind == 2:
        assert raw == final and rank(gt_pair) > 1 and final not in targets
    return {
        'pair_id': pid, 'house': row['house_id'], 'floor': row['floor_id'],
        'room_a': record['room_a'], 'room_b': record['room_b'],
        'pano_a': record['pano_a'], 'pano_b': record['pano_b'], 'split': 'val',
        'gt_positive': positive, 'candidate_recalled': bool(row['portallayout.evaluation']['gt_candidate_recalled']) if positive else None,
        'category': kind, 'candidate_counts': [na, nb], 'raw_rank1': raw,
        'raw_best_score': float(scores[tuple(raw)]), 'final_selection': final,
        'accepted': final != [-1, -1], 'gt_pairs': targets, 'gt_pair_displayed': gt_pair,
        'gt_rank': rank(gt_pair) if positive else None,
        'gt_score': float(scores[tuple(gt_pair)]) if positive else None,
        'pair_match_probability': f['pair_match_probability'],
        'pair_no_match_probability': f['pair_no_match_probability'],
        'dustbin_logit': db, 'raw_rank1_logit': float(logits[tuple(raw)]),
        'raw_rank1_to_dustbin_logit_margin': float(logits[tuple(raw)] - db),
        'dustbin_source': 'direct per-pair evidence' if direct else 'validated shared serialized constant; local padding fields unusable',
        'score_source': str(frozen_map[pid]), 'logit_source': str(score_map[pid]),
        'max_score_difference_to_rescoring': float(np.abs(scores-rescored).max()),
        'score_semantics': 'candidate_pair_score; not calibrated adjacency probability',
        'raw_rank_definition': 'stable descending real-real candidate_pair_score; no dustbin',
        'images': images, 'image_paths': image_paths, 'intervals': intervals,
        'gt_intervals': gt_intervals, 'scores': scores.tolist(),
    }


def panorama(ax, case, side, fontsize=12, detail=False):
    image = case['images'][side]
    height = 128  # Longitude/pixel coordinates normalized to a true 2:1 panorama.
    if detail:
        pixels=np.asarray(image)
        ax.imshow(np.concatenate([pixels]*3,axis=1),extent=(-256,512,height,0),aspect='auto')
    else:
        ax.imshow(image, extent=(0, 256, height, 0), aspect='auto')
    indices = case['intervals'][side]
    selected = case['final_selection'][side] if case['accepted'] else None
    focus = interval_center(indices[selected if selected is not None else case['raw_rank1'][side]],256)
    for i, interval in enumerate(indices):
        color = SELECTED if i == selected else CAND
        spans=[(a+shift,b+shift) for a,b in interval_pixel_spans(interval,256,256)
               for shift in ([-256,0,256] if detail else [0])]
        for start, end in spans:
            ax.add_patch(Rectangle((start, 0), end-start, height, facecolor=color,
                                  alpha=.09, edgecolor='none'))
            ax.add_patch(Rectangle((start, 0), end-start, height, fill=False,
                                  edgecolor=color, linewidth=2.3 if i == selected else 1.2))
        center = interval_center(interval, 256)
        for shift in ([-256,0,256] if detail else [0]):
            cx=center+shift if detail else max(8,min(248,center))
            if detail and not focus-42 <= cx <= focus+42:
                continue  # Omit off-crop labels rather than drawing truncated glyphs.
            ax.text(cx, (58 if detail else height*(.06 if i%2 == 0 else .19)), ('A' if side == 0 else 'B')+str(i),
                    ha='center', va='top', fontsize=fontsize, color='white', weight='bold',clip_on=True,
                    bbox={'facecolor':color, 'edgecolor':'white', 'linewidth':.5, 'pad':2})
    if case['gt_positive']:
        spans=[(a+shift,b+shift) for a,b in interval_pixel_spans(case['gt_intervals'][side],256,256)
               for shift in ([-256,0,256] if detail else [0])]
        for start, end in spans:
            ax.add_patch(Rectangle((start, height*.015), end-start, height*.97,
                                  fill=False, edgecolor=GT, linewidth=3, linestyle='--'))
            ax.plot([start,end],[height*.97]*2,color=GT,linewidth=5)
    if detail:
        chosen = selected if selected is not None else case['raw_rank1'][side]
        center = interval_center(indices[chosen], 256)
        ax.set_xlim(center-48,center+48)
        ax.set_ylim(95.4,54)
    else:
        ax.set_xlim(0,256); ax.set_ylim(height,0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#d7dde3')


def matrix(ax, case, fontsize=13):
    scores = np.array(case['scores'])
    ax.imshow(scores, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    na, nb = scores.shape
    ax.set_xticks(range(nb));ax.set_xticklabels([f'B{i}' for i in range(nb)],fontsize=fontsize)
    ax.set_yticks(range(na));ax.set_yticklabels([f'A{i}' for i in range(na)],fontsize=fontsize)
    ax.tick_params(length=0, pad=4)
    for i in range(na):
        for j in range(nb):
            ax.text(j,i,f'{scores[i,j]:.3f}',ha='center',va='center',fontsize=fontsize,
                    color='white' if scores[i,j]>.55 else '#152b43',weight='bold')
    for ij in case['gt_pairs']:
        i,j=ij;ax.add_patch(Rectangle((j-.46,i-.46),.92,.92,fill=False,edgecolor=GT,linewidth=2.8,linestyle='--'))
    i,j=case['raw_rank1']
    ax.add_patch(Rectangle((j-.40,i-.40),.80,.80,fill=False,edgecolor=RAW,linewidth=2))
    if case['accepted']:
        i,j=case['final_selection'];ax.add_patch(Rectangle((j-.31,i-.31),.62,.62,fill=False,edgecolor=SELECTED,linewidth=2.3))
    ax.set_xlim(-.5,nb-.5);ax.set_ylim(na-.5,-.5)
    for spine in ax.spines.values():spine.set_visible(False)


def render_case(case, path):
    kind = case['category'];color=COLORS[kind]
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.035,.945,f'{kind+1:02d}  {TITLES[kind]}',fontsize=23,weight='bold',color='#173047')
    fig.text(.035,.905,'T3 · ZInD validation · qualitative case',fontsize=12,color='#586979')
    for side,x in [(0,.035),(1,.665)]:
        letter='A' if side==0 else 'B'
        fig.text(x,.84,f'Room {letter}',fontsize=17,weight='bold',color='#173047')
        ax=fig.add_axes([x,.46,.30,.2666667]);panorama(ax,case,side)
        ax=fig.add_axes([x,.15,.30,.23]);panorama(ax,case,side,detail=True)
        selected=case['final_selection'][side] if case['accepted'] else case['raw_rank1'][side]
        fig.text(x,.40,f'Detail: {letter}{selected} · '+('final selection' if case['accepted'] else 'raw Rank-1'),fontsize=12,color='#586979')
    fig.text(.50,.84,'Matcher decision',ha='center',fontsize=17,weight='bold',color='#173047')
    fig.text(.50,.775,'Real-real candidate score',ha='center',fontsize=12,color='#586979')
    matrix(fig.add_axes([.398,.485,.205,.25]),case)
    fig.text(.50,.423,f"Raw Rank-1: {label(case['raw_rank1'])}  ·  {case['raw_best_score']:.3f}",ha='center',fontsize=13,color=RAW,weight='bold')
    if case['gt_positive']:
        gtline=f"GT recalled: {label(case['gt_pair_displayed'])}\nGT score {case['gt_score']:.3f} · rank {case['gt_rank']}"
    else:gtline='GT label: Non-adjacent\nNo shared opening'
    fig.text(.50,.347,gtline,ha='center',va='center',fontsize=13,color=GT,linespacing=1.6)
    final='ACCEPT  '+label(case['final_selection']) if case['accepted'] else 'REJECT  ·  No Match'
    fig.text(.50,.247,final,ha='center',va='center',fontsize=17,weight='bold',color='white',
             bbox={'boxstyle':'round,pad=.55','facecolor':color,'edgecolor':'none'})
    fig.text(.50,.165,('Passes' if case['accepted'] else 'Rejected by')+' bidirectional dustbin\n'+f"No-match score: {case['pair_no_match_probability']:.3f}",ha='center',va='center',fontsize=11.5,color='#586979',linespacing=1.6)
    fig.text(.035,.072,'Blue: predicted candidates     Green dashed: GT     Purple: raw Rank-1     Orange: final selection',fontsize=11.5,color='#34485a')
    fig.text(.035,.033,FOOTNOTE,fontsize=11.5,color='#586979')
    for ext in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=180,facecolor='white')
    return fig


def render_montage(cases, output):
    # Native five-row 16:9 layout, not a shrunken collage of debug screenshots.
    fig=plt.figure(figsize=(24,13.5),facecolor='white')
    fig.text(.025,.965,'T3 Matcher | Five validation cases',fontsize=27,weight='bold',color='#173047')
    fig.text(.025,.932,'Predicted candidates → real-real ranking → final dustbin decision',fontsize=17,color='#586979')
    for i,c in enumerate(cases):
        y=.745-i*.171
        # Full panoramas preserve their 2:1 ratio. Middle columns carry decisions.
        panorama(fig.add_axes([.025,y,.16,.142]),c,0,fontsize=14)
        panorama(fig.add_axes([.815,y,.16,.142]),c,1,fontsize=14)
        fig.text(.025,y+.148,'Room A',fontsize=13,color='#586979')
        fig.text(.815,y+.148,'Room B',fontsize=13,color='#586979')
        fig.text(.212,y+.118,f'{i+1:02d}  '+['Correct Match','Recalled / Dustbin Reject','Recalled / Wrong Selection','Negative / Correct Reject','Negative / False Accept'][i],fontsize=20,weight='bold',color=COLORS[i])
        fig.text(.212,y+.082,f"Raw #1: {label(c['raw_rank1'])} · {c['raw_best_score']:.3f}",fontsize=17,color=RAW)
        g=f"GT: {label(c['gt_pair_displayed'])} · rank {c['gt_rank']} · {c['gt_score']:.3f}" if c['gt_positive'] else 'GT: Non-adjacent · no shared opening'
        fig.text(.212,y+.045,g,fontsize=16,color=GT)
        final='ACCEPT '+label(c['final_selection']) if c['accepted'] else 'REJECT · No Match'
        fig.text(.212,y+.008,final+f"   |   no-match {c['pair_no_match_probability']:.3f}",fontsize=17,weight='bold',color=COLORS[i])
        matrix(fig.add_axes([.636,y+.012,.15,.112]),c,fontsize=15)
    fig.text(.025,.035,'Blue: candidates   Green dashed: GT   Purple: raw Rank-1   Orange: final selection',fontsize=16,color='#34485a')
    fig.text(.025,.008,FOOTNOTE,fontsize=16,color='#586979')
    for ext in ['png','pdf','svg']:fig.savefig(output/f't3_matcher_five_cases.{ext}',dpi=180)
    plt.close(fig)


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    rows=read(F/'per_pair.jsonl',True); byid={r['pair_id']:r for r in rows}
    assert len(rows)==len(byid)==3630
    manifest=read(ROOT/'data/ZInD_matching/val.json');assert manifest['split']=='val'
    records={r['id']:r for r in manifest['pairs']}; assert set(records)==set(byid)
    metadata=read(F/'run_metadata.json');assert metadata['split']=='val'
    read(F/'summary.json')
    vm=read(V/'visualization_manifest.json')
    observations={o['pair_id']:o for o in read(V/'visual_observations.jsonl',True)}
    audits={r['pair_id']:r for r in read(A/'per_pair_val.jsonl',True)}
    frozen_map={p.name.split('_',1)[1][:-8]:p for p in (F/'frozen_predictions').glob('*.json.gz')}
    score_map={}
    for root in [A/'pair_scores/val_O0',B/'negative_pair_scores/val']:
        for p in root.rglob('*.json.gz'):
            # Score filenames may be hashes; only read val records to build identity map.
            d=read(p);score_map[d['pair_id']]=p
    selections=[];chosen=[];verification=[]
    for kind in range(5):
        candidates=[r for r in rows if category(r)==kind]
        if kind<3:
            reviewed={x['pair_id'] for x in vm['selected']}
            candidates=[r for r in candidates if r['pair_id'] in reviewed and observations[r['pair_id']]['gt_opening_visibility']=='clear']
            candidates.sort(key=lambda r:(observations[r['pair_id']]['annotation_ambiguity']!='none',
                r['prediction']['candidate_count_A']*r['prediction']['candidate_count_B'],r['pair_id']))
        else:
            candidates=[r for r in candidates if 2<=r['prediction']['candidate_count_A']<=3 and 2<=r['prediction']['candidate_count_B']<=3]
            candidates.sort(key=lambda r:(r['prediction']['candidate_count_A']*r['prediction']['candidate_count_B'],r['pair_id']))
        pool=[];houses=set()
        for row in candidates:
            if row['house_id'] in houses:continue
            case=load_case(row,records[row['pair_id']],frozen_map,score_map,audits,Path(manifest['dataRoot']))
            if max(case['candidate_counts'])>5:continue
            houses.add(row['house_id']);pool.append(case)
            if len(pool)==3:break
        assert len(pool)==3, (kind,len(pool))
        # Deterministic primary: clearest existing audit annotation first, then
        # smallest matrix, then pair ID. No score maximization or GT-based reranking.
        primary = min(range(3),key=lambda j:np.prod(pool[j]['candidate_counts'])) if kind==2 else 0
        for j,c in enumerate(pool):
            c['main_case']=j==primary
            c['selection_reason']=('Existing visual audit: clear GT; prefer no annotation ambiguity, then small matrix and stable ID.'
                if kind<3 else 'Unambiguous frozen negative label/decision; 2–3 candidates per side; smallest matrix then stable ID; distinct-house alternatives.')
            if kind<3:
                c['prior_visual_observation']={k:observations[c['pair_id']][k] for k in ['gt_opening_visibility','annotation_ambiguity','occlusion']}
            c['main_selection_reason']=('Smallest matrix among three visually clear wrong-selection examples; any prior annotation ambiguity is retained in metadata.'
                 if kind==2 else 'First valid candidate under the stated deterministic protocol.')
            review=output/'candidate_review';review.mkdir(exist_ok=True)
            fig=render_case(c,review/f'case_{kind+1:02d}_option_{j+1}');plt.close(fig)
            selections.append({k:v for k,v in c.items() if k!='images'})
            verification.append({'pair_id':c['pair_id'],'category':kind+1,'split':'val',
                'classification_verified':True,'gt_intervals_match_manifest':True,
                'negative_has_no_gt_correspondence':not c['gt_pairs'] if kind>=3 else None,
                'raw_rank1_matches_frozen_scores_and_rescoring':True,
                'final_selection_matches_frozen_and_logit_gate':True,
                'rgb_decoded':True,'test_data_accessed':False})
        chosen.append(pool[primary])
    with PdfPages(output/'t3_matcher_case_slides.pdf') as pdf:
        for kind,c in enumerate(chosen):
            fig=render_case(c,output/f'case_{kind+1:02d}_{NAMES[kind]}');pdf.savefig(fig);plt.close(fig)
    render_montage(chosen,output)
    figure_checks=[]
    for p in sorted(output.glob('*')):
        if p.suffix=='.png':
            with Image.open(p) as im:
                im.load();size=im.size
            assert abs(size[0]/size[1]-16/9)<1e-6
            figure_checks.append({'file':p.name,'decoded':True,'size':size})
        elif p.suffix=='.svg':
            tree=ET.parse(p)
            images=tree.findall('.//{http://www.w3.org/2000/svg}image')
            assert images and all(x.attrib.get('{http://www.w3.org/1999/xlink}href','').startswith('data:') for x in images)
            figure_checks.append({'file':p.name,'xml_valid':True,'rgb_embedded':True})
    for path,digest in HASHES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    payload={'selection_protocol':'Qualitative selection only. Reviewed positive clarity/ambiguity; bounded matrix; deterministic ID; distinct houses within each category.',
             'checkpoint':metadata['checkpoint'],'split':'val','candidates':selections,
             'main_pair_ids':[c['pair_id'] for c in chosen],'input_sha256':HASHES}
    (output/'case_selection.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    checks={'passed':True,'case_count':5,'candidate_count':15,'checks':verification,
            'figure_checks':figure_checks,
            'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_hashes_unchanged':True,'prediction_modified':False,
            'training_run':False,'gpu_forward_run':False,'test_data_accessed':False,
            'limitations':['Selected qualitative examples, not an aggregate estimate.',
                'Match/no-match scores are not assumed complementary or calibrated.',
                'Raw Rank-1 uses candidate_pair_score, not bare assignment-logit rank.']}
    (output/'verification.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
    lines=['# T3 Matcher five validation cases', '',
           'All figures use existing frozen validation predictions and RGB. No training, model loading, GPU forward or test access.', '',
           'Use t3_matcher_five_cases.png/pdf/svg for the 16:9 overview, or t3_matcher_case_slides.pdf for five full-size pages. Individual cases also have PNG/PDF/SVG versions.', '',
           'Full panoramas preserve their 2:1 ratio; detail crops are centered on the predicted final choice (raw Rank-1 when rejected), with cyclic display across the seam. GT never determines the crop or prediction.', '',
           'No-match scores and real-real scores are not complementary calibrated probabilities. Acceptance is the original bidirectional dustbin decision, not a threshold on the displayed no-match scalar.', '',
           '| Case | Pair ID | Raw #1 | Final | GT rank |', '| --- | --- | --- | --- | --- |']
    for c in chosen:
        final=label(c['final_selection']) if c['accepted'] else 'REJECT / No Match'
        lines.append(f"| {NAMES[c['category']]} | {c['pair_id']} | {label(c['raw_rank1'])} | {final} | {c['gt_rank']} |")
    lines.extend(['', 'case_selection.json contains all three alternatives per category, selection reasons, prior positive visual-review uncertainty, and source SHA-256 values. candidate_review/ contains their rendered panels.', '',
                  'These are selected qualitative examples, not aggregate performance or causal proof. Wrong-selection case 0220 retains the prior audit flag of possible annotation ambiguity; its frozen classification is unambiguous under the existing evaluation protocol.', '',
                  'Run: `env CUDA_VISIBLE_DEVICES=\'\' PYTHONDONTWRITEBYTECODE=1 /home/feixia/anaconda3/envs/bi_layout/bin/python -B tools/visualize_zind_matcher_five_cases.py`'])
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:c[k] for k in ['pair_id','house','floor','room_a','room_b','pano_a','pano_b','raw_rank1','final_selection','gt_rank']} for c in chosen],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir',type=Path,default=ROOT/'outputs/zind_matcher_visual_report')
    main(parser.parse_args().output_dir.resolve())
