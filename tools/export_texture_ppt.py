"""Export editable, data-derived PPT figures without changing source photographs.

Requires python-pptx. Reference polygons are annotation projections, not predictions.
"""
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
INK = '172B45'
GREEN = '059669'
BLUE = '00A8E8'
RED = 'B34336'


def text(slide, x, y, w, h, value, size=18, color=INK, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(value.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.name = 'Noto Sans CJK SC'; p.font.size = Pt(size)
        p.font.bold = bold; p.font.color.rgb = RGBColor.from_string(color)
        p.space_after = Pt(5)
    return box


def rect(slide, x, y, w, h, color):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(color); s.line.fill.background()
    return s


def line(slide, x1, y1, x2, y2, color, width=2):
    s = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    s.line.color.rgb = RGBColor.from_string(color); s.line.width = Pt(width)


def segments(start, length):
    start %= 1
    return [(start, min(1, start+length))] + ([(0, start+length-1)] if start+length > 1 else [])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    (out/'originals').mkdir(exist_ok=True)
    run = ROOT/'experiments/texture_ambiguity_20261029'
    report = json.loads((run/'paired_probe.json').read_text())
    all_pairs = {x['pair_id']:x for x in json.loads((run/'all_pairs.json').read_text())}
    contract = json.loads((ROOT/'experiments/research1_round2_20261024/run_contract.json').read_text())
    weak = json.loads((ROOT/contract['weak_supervision']).read_text())
    plan = [
        ('0561', [33,34], '主展示 01', '浅色天花板与暖色地面', '地面色泽、局部纹理有差异；非严格同材质对照。'),
        ('0988', [9,11], '主展示 02', '浅色天花板与木纹地板', '另有一个未知匹配；不能称整对完全正确。'),
        ('0988', [9,10], '补充：相关视角', '同一房屋的另一视角', '与主展示 02 共享全景 9，不是独立案例。'),
        ('0060', [26,29], '补充：混合材质', '木色地面与地毯同时可见', '纹理条件不严格，仅作为判断差异的补充。'),
        ('0726', [7,27], '反例 01', 'SALVe 适配版也接受了正确对齐', '保留反例；相似外观并不必然导致基线失败。'),
        ('0430', [5,6], '反例 02', 'SALVe 适配版也接受了正确对齐', '浅色天花板与方砖地面；不能用于支持基线失败。'),
    ]
    prs = Presentation(); prs.slide_width = Inches(16); prs.slide_height = Inches(9)
    manifest = {'source_report_sha256':hashlib.sha256((run/'paired_probe.json').read_bytes()).hexdigest(),
                'reference_outline':'dataset annotation projection, display only',
                'prediction_marker':'frozen R3 detected opening horizontal angular interval',
                'human_verified':False,'slides':[]}
    for number, (building, panos, role, title, caution) in enumerate(plan,1):
        c = next(x for x in report['cases'] if x['building']==building and x['panos']==panos)
        pair = all_pairs[c['pair_id']]
        matches = [m for m in pair['matches'] if m['target']==1]
        assert len(matches)==1, 'This figure template expects one weak-positive match'
        m = matches[0]
        annotations = {d['observation_id']:d for d in map(json.loads,
            (ROOT/contract['portallayout.datasets']/'labels'/pair['floor_uid']/'observations_gt.jsonl').read_text().splitlines())}
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid(); slide.background.fill.fore_color.rgb = RGBColor(255,255,255)
        text(slide,.35,.15,15.2,.45,f'{role}  |  {title}',27,bold=True)
        side_manifest=[]
        for idx,(uid,source) in enumerate(zip(pair['observation_ids'],pair['images'])):
            meta_path=next((ROOT/contract['prediction_cache']/'observations'/uid).glob('*.json'))
            meta=json.loads(meta_path.read_text()); ci=m['index_a' if idx==0 else 'index_b']
            opening=meta['prediction']['openings'][ci]
            assert opening['candidate_id']==m['candidate_id_a' if idx==0 else 'candidate_id_b']
            association=weak['associations'][uid][ci]
            assert association['status']=='associated'
            ref=next(z for z in annotations[uid]['interfaces_by_layout']['layout_raw']
                     if z['interface_observation_id']==association['interface_id'])
            proj=ref['projection']; assert proj and proj['length_u']>0
            src=Path(source); destination=out/'originals'/f'{building}_{src.name}'
            if not destination.exists(): shutil.copyfile(src,destination)
            assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(destination.read_bytes()).digest()
            x,y,w,h=.35,.75+idx*4.0,7.8,3.9
            slide.shapes.add_picture(str(src),Inches(x),Inches(y),width=Inches(w),height=Inches(h))
            pano=int(re.search(r'_pano_(\d+)\.jpg$',src.name).group(1))
            rect(slide,x,y,w,.27,INK)
            text(slide,x+.08,y+.025,w-.16,.22,f'{"A" if idx==0 else "B"}  |  房屋 {building} · 全景 {pano}  |  对应 P1',12,'FFFFFF')
            # Ground-truth projection outline: retain the original spherical curve and seam.
            us=proj['sample_u']; top=proj['top_v']; bottom=proj['bottom_v']
            for vs in (top,bottom):
                for j in range(len(us)-1):
                    if abs(us[j+1]-us[j])<.5:
                        line(slide,x+us[j]*w,y+vs[j]*h,x+us[j+1]*w,y+vs[j+1]*h,BLUE,2.2)
            for j in (0,-1):line(slide,x+us[j]*w,y+top[j]*h,x+us[j]*w,y+bottom[j]*h,BLUE,2.2)
            n=meta['tensor_spec']['depth_enclosed']['shape'][0]
            start,end=opening['token_interval_inclusive']; length=((end-start)%n+1)/n
            for a,b in segments(start/n,length):
                yy=y+.40
                line(slide,x+a*w,yy,x+b*w,yy,GREEN,3.5)
                for xx in (x+a*w,x+b*w):line(slide,xx,yy-.06,xx,yy+.07,GREEN,2.5)
            mid=(start/n+length/2)%1
            xx=min(x+w-.65,max(x+.03,x+mid*w-.27))
            rect(slide,xx,y+.48,.56,.25,GREEN);text(slide,xx+.08,y+.49,.48,.23,'P1',13,'FFFFFF',True)
            side_manifest.append(dict(observation_id=uid,image=str(src),copied_image=str(destination.relative_to(out)),
                sha256=hashlib.sha256(src.read_bytes()).hexdigest(),candidate_id=opening['candidate_id'],
                candidate_index=ci,token_count=n,token_interval_inclusive=[start,end],
                reference_interface_id=ref['interface_observation_id'],reference_projection=proj))
        x=8.55; ww=7.0
        text(slide,x,.8,ww,.4,f'房屋 {building}  ·  全景 {panos[0]} ↔ {panos[1]}',23,bold=True)
        rect(slide,x,1.37,ww,1.65,'EDF8F3')
        text(slide,x+.2,1.5,ww-.4,.35,'本项目 R3  |  开口对应',22,GREEN,True)
        text(slide,x+.2,1.98,ww-.4,.45,'命中自动弱标签正对应',25,GREEN,True)
        text(slide,x+.2,2.53,ww-.4,.36,f'匹配分数 {m["confidence"]:.3f}   ·   未知匹配 {c["r3_unknown_matches"]} 个',18)
        accepted=c['salve_correct_accepted']>0
        rect(slide,x,3.25,ww,1.85,'F3F5F8' if accepted else 'FCF2EF')
        text(slide,x+.2,3.38,ww-.4,.35,'SALVe-DAV2 适配版  |  对齐验证',21,INK,True)
        text(slide,x+.2,3.86,ww-.4,.4,'接受正确对齐（反例）' if accepted else '拒绝正确对齐候选',25,INK if accepted else RED,True)
        cmp='≥' if accepted else '<'
        text(slide,x+.2,4.38,ww-.4,.3,f'正确候选最高分 {c["salve_max_correct_score"]:.3f} {cmp} 固定阈值 0.93',17)
        text(slide,x+.2,4.75,ww-.4,.3,f'接受正确 / 错误假设：{c["salve_correct_accepted"]} / {c["salve_wrong_accepted"]}',17)
        text(slide,x,5.45,ww,.3,'图中标签',20,bold=True)
        text(slide,x,5.92,ww,.4,'绿色 P1 / 横线：模型预测的开口角度范围',17,GREEN)
        text(slide,x,6.33,ww,.4,'蓝色轮廓：数据集参考开口投影（仅展示）',17,BLUE)
        text(slide,x,6.78,ww,.7,'P1 表示本例选中的一组弱对应；\n蓝色轮廓不是模型预测，尚未人工确认。',17)
        text(slide,x,7.65,ww,.7,caution,18,RED)
        text(slide,.35,8.78,15.3,.18,'探索性开发集案例 · 两模型分数不可直接比大小 · 基线使用 DAV2 替代深度 · 未证明纹理因果关系或正确位姿重建',10,'526176')
        slide.notes_slide.notes_text_frame.text=(f'案例 {c["pair_id"]}。'+caution+
            '\n本例按R3弱标签命中筛选，不能作为无偏性能估计。SALVe分数判断对齐，R3分数判断开口对应。'+
            '\n原始分数与图像来源见同目录 manifest.json；参考轮廓仅用于解释，没有作为新增推理输入。')
        manifest['slides'].append(dict(slide=number,role=role,case=c,pair_id=c['pair_id'],sides=side_manifest))
    prs.save(out/'纹理相似房间对_可编辑标注.pptx')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    readme=out/'README.md'
    history=readme.read_text().split('\n\n',1)[0]+'\n' if readme.exists() else ''
    readme.write_text(history+f'修改记录：{now}（Asia/Shanghai）—导出PPT案例、可编辑预测范围和参考标注，保留补充及反例。\n\n'
        '# PPT 图片使用说明\n\n'
        '主展示：第1页 0561（33↔34）、第2页 0988（9↔11）。第3页是相关视角，第4页是混合材质补充，第5、6页是反例。\n\n'
        '绿色 P1 和横线来自模型的开口水平角度范围；蓝色轮廓来自数据集参考投影，仅用于展示，不能当作预测。'
        '全景原图不做修补或生成，originals 是逐字节一致的原图副本。PPTX 中图片、文字、线条均独立，可修改。\n\n'
        '可用于PPT的表述：在所选开发集案例中，R3命中自动弱标签中的开口对应，而SALVe-DAV2适配版拒绝了正确对齐候选。'
        '这提供局部对应识别的定性支持，尚不能证明整体方法优于原版SALVe，也不能证明纹理相似是失败原因。\n\n'
        '不要写成：纹理相似使SALVe必然失败；本方法已经正确重建这些房间；图中蓝色开口是模型检测结果。'
        '0988两对共享一张全景且各有一个未知匹配；0060含混合材质；0561地面色泽和局部纹理有差异。\n\n'
        '每页原始分数、候选ID、标注投影、图像哈希见 manifest.json。结果来自既有真实推理，本次只制图，没有重跑或调参。\n')
    print(out)


if __name__=='__main__':main()
