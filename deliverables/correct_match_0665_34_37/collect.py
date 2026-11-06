from pathlib import Path
import json,shutil,hashlib,xml.etree.ElementTree as E,zipfile,html
O=Path(__file__).resolve().parent;R=O.parents[1]
B=R/'src/output/zind_room_group_debug/t3_best_0665_floor03_candidate_local_m2'
base='0665_floor_03_pano_34_pano_37_3'
d=json.loads((B/'pairs'/f'{base}.json').read_text());p=d['prediction'];ev=d['evaluation']
assert p['best_candidate_pair']==[1,0] and p['is_match'] and ev['classification_correct'] and ev['gt_candidate_recalled'] and ev['gt_candidate_rank']==1
assert p['candidate_pair_scores'][1][0]==max(max(row) for row in p['candidate_pair_scores'])
manifest={}
def cp(a,n):
 a=Path(a);shutil.copy2(a,O/n);manifest[n]={'source':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest()}
def dump(n,v): (O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
for side,r,num in [('A','01',34),('B','04',37)]:
 room=json.loads((B/'rooms'/f'complete_room_{r}.json').read_text());cp(room['image_path'],f'{side}_pano_{num}.jpg');cp(B/'rooms'/f'complete_room_{r}.json',f'{side}_room_original_with_GT.json')
 est={'room_id':room['room_id'],'pano_id':room['pano_id'],'single_room_layout':room['single_room_layout'],'opening_prediction':room['opening_prediction'],'gt_used':False};dump(f'{side}_prediction.json',est)
 lay=room['single_room_layout'];polys=[]
 for key in ['layout_polygon_pre_manhattan_local_bi_xz']:
  polys.append([(-a,b) for a,b in lay[key]])
 polys.append([(-a,b) for a,b in lay['extended_postprocess']['layoutPolygonPreManhattanLocalBiXZ']])
 portal=p[f'portal_geometry_{side}']['endpoints_local'];allpts=sum(polys,[])+portal+[[0,0]]
 xmin=min(a for a,b in allpts);xmax=max(a for a,b in allpts);ymin=min(b for a,b in allpts);ymax=max(b for a,b in allpts);sc=min(650/(xmax-xmin),550/(ymax-ymin))
 def xy(a,b):return (450+(a-(xmin+xmax)/2)*sc,360-(b-(ymin+ymax)/2)*sc)
 parts=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="720" viewBox="0 0 900 720"><rect width="900" height="720" fill="white"/>']
 for pts,col in zip(polys,['#1a73e8','#e69500']):parts.append('<polygon points="'+' '.join('%.3f,%.3f'%xy(a,b) for a,b in pts)+'" fill="none" stroke="'+col+'" stroke-width="3"/>')
 a,b=xy(*portal[0]),xy(*portal[1]);parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#a61e24" stroke-width="6"/>');cx,cy=xy(0,0);parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="black"/>')
 parts.append(f'<text x="25" y="30" font-size="22">{side}: pano {num} / room {r} / predicted pre-Manhattan</text>')
 for yy,st,col in [(650,'Blue: enclosed; Orange: extended','#333'),(681,'Red: selected predicted portal; Black: camera; no GT','#333')]:parts.append(f'<text x="25" y="{yy}" font-size="20" fill="{col}">{st}</text>')
 parts.append('</svg>');(O/f'{side}_estimated_layout.svg').write_text('\n'.join(parts))
cp(B/'pairs'/f'{base}.json','pair_original_with_GT.json')
cp(B/'pairs'/f'{base}_pre_manhattan_joint.png','joint_layout_original.png')
cp(B/'pairs'/f'{base}_pre_manhattan_joint.svg','joint_layout_original.svg')
cp(B/'pairs'/f'{base}_frame_preserving_joint.png','joint_layout_manhattan_original.png')
cp(B/'pairs'/f'{base}_independent.svg','before_alignment.svg')
sr=E.parse(B/'pairs'/f'{base}_pre_manhattan_joint.svg').getroot()
for e in list(sr):
 if e.tag.endswith('text'):sr.remove(e)
for x,y,t,col in [(30,30,'T3 / train / 0665 / floor_03 / pano 34 <- 37','#333'),(280,340,'Room 01','#1e6ecd'),(30,875,'Green: Room 04','#289128'),(420,875,'Red: shared predicted opening','#dc1e1e')]:
 e=E.SubElement(sr,'{http://www.w3.org/2000/svg}text',{'x':str(x),'y':str(y),'font-size':'24','fill':col});e.text=t
E.ElementTree(sr).write(O/'joint_layout_clean.svg',encoding='utf-8',xml_declaration=True)
dump('pair_prediction_only.json',p)
summary={'pair_id':base,'model':'T3 best.pt, completed epoch 8','split':'ZInD official train; qualitative/debug only','selected_predicted_candidates':[1,0],'correct_match_evidence':{k:ev[k] for k in ['classification_correct','gt_candidate_rank','gt_candidate_recalled']},'candidate_score':p['pair_match_probability'],'score_is_calibrated_probability':False,'estimated_B_to_A':p['relative_pose'],'evaluation':ev,'gt_usage':'post-inference evaluation only','not_latest_ObservablePairs_model':True}
dump('summary.json',summary);dump('source_manifest.json',manifest)
readme='''# 正确匹配全景图对：0665 / floor_03 / pano 34 与 37

- A：房间01，全景34；B：房间04，全景37。
- 历史T3 best.pt，completed epoch 8；官方train上的定性案例，并非最新ObservablePairs模型或独立测试结果。
- 预测候选A1 ↔ B0，接受匹配；对应分数0.881621（非校准概率）。已有评估：正确候选已召回，GT候选排名1，正负分类正确；所选候选也是实际分数第一。
- GT标注候选索引是A2 ↔ B0，和预测候选A1 ↔ B0属于不同候选集合，索引不能直接比较。正确性按原评估的区间对应映射判定。

## 文件

直接打开index.html可查看全部图片。

- A_pano_34.jpg、B_pano_37.jpg：原始全景，不裁剪、不拉伸。
- A/B_estimated_layout.svg：从已有JSON重新绘制单图预测；蓝色闭合布局、橙色外延布局、红色选中开口。使用pre-Manhattan预测坐标、等比例绘制；不含GT。不将两张独立图的显示缩放解释为相对尺度。
- A/B_prediction.json：完整单图估计及开口概率、候选信息，不含GT。
- joint_layout_clean.svg：已有联合布局SVG的相同多边形、接口与相机标记，仅移除重叠英文标签并重排说明。
- joint_layout_original.png/svg：未改动的原始联合布局；joint_layout_manhattan_original.png为另一种后处理表示。
- pair_prediction_only.json：匹配、B→A变换、预测联合布局等全部模型输出。
- 带original_with_GT字样的JSON含离线GT对照，仅用于核验。

## 对齐结果与边界

B→A旋转约−173.178°，平移[−1.042097, −0.673623]（A局部坐标，不能直接写成米），相对尺度0.780597。位姿由预测开口的candidate-local M2几何得到，非GT对齐。

已有后验评价：旋转误差4.356°；平移误差0.600936米（仅作为该历史记录已有米制评价）；绝对log尺度误差0.247696。匹配正确并不表示拼接无误差。单图与联合图均为估计结果，没有进行新训练、模型推理或用GT替换预测。
'''
(O/'README.md').write_text(readme)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>正确匹配：0665 全景34与37</title><style>body{font-family:sans-serif;margin:30px;background:#f4f4f4;color:#222}h1{color:#a61e24}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0;background:white;padding:15px}img{width:100%;height:auto}figcaption{padding:12px}section{max-width:1000px;margin:25px auto;background:white;padding:20px}p{line-height:1.7}</style><h1>正确匹配案例：全景34 ↔ 全景37</h1><p>T3 · ZInD train · house 0665 · floor_03。预测 A1–B0，正确候选排名1，分数0.881621（非校准概率）。GT仅用于事后核验；几何仍存在误差。</p><main>'''
for f,t in [('A_pano_34.jpg','A：全景34，房间01'),('B_pano_37.jpg','B：全景37，房间04'),('A_estimated_layout.svg','A单图估计：闭合/外延布局及选中开口'),('B_estimated_layout.svg','B单图估计：闭合/外延布局及选中开口')]:page+=f'<figure><a href="{f}"><img src="{f}"></a><figcaption>{t}</figcaption></figure>'
page+='</main><section><h2>预测联合布局（B→A）</h2><img src="joint_layout_clean.svg"><p>预测旋转 −173.178°；相对尺度0.780597。旋转误差4.356°，已有记录米制平移误差0.601米。轮廓来自原始输出，仅调整标签位置。</p><a href="summary.json">结果摘要</a> · <a href="README.md">文件与口径说明</a> · <a href="pair_original_with_GT.json">原始结果及GT核验</a></section></html>'
(O/'index.html').write_text(page)
with zipfile.ZipFile(O.parent/'正确匹配案例_0665_34_37.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in O.iterdir():
  if f.suffix!='.py':z.write(f,'正确匹配案例_0665_34_37/'+f.name)
print(O)
