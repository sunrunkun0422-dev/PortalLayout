import sys,json,copy,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,'/tmp/bi_layout_ppt_deps')
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.xmlchemy import OxmlElement
O=Path(__file__).resolve().parent;R=O.parents[1];D=R/'deliverables/correct_match_0665_34_37'
prs=Presentation(R/'deliverables/opening_defense/孙润坤_硕士开题汇报.pptx')
layout=prs.slides[1].slide_layout
for sid in list(prs.slides._sldIdLst):prs.part.drop_rel(sid.rId);prs.slides._sldIdLst.remove(sid)
s=prs.slides.add_slide(layout)
for sh in list(s.shapes):s.shapes._spTree.remove(sh._element)
WHITE='FFFFFF';GRAY='5F6368';BLUE='2475EA';RED='A61E24';GREEN='418358';DARK='202124';ORANGE='AC6B22'
def text(x,y,w,h,t,size=16,c=DARK,b=False,center=False):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
 tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=Pt(0)
 for i,st in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=st;p.font.name='Microsoft YaHei';p.font.size=Pt(size);p.font.bold=b;p.font.color.rgb=RGBColor.from_string(c);p.space_after=Pt(3)
  if center:p.alignment=PP_ALIGN.CENTER
  for r in p.runs:
   rp=r._r.get_or_add_rPr()
   for tag in ['a:latin','a:ea','a:cs']:
    e=OxmlElement(tag);e.set('typeface','Microsoft YaHei');rp.append(e)
 return sh

def box(x,y,w,h,fill='E8F0FE',stroke=BLUE,dash=False,round=True):
 sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if round else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill);sh.line.color.rgb=RGBColor.from_string(stroke);sh.line.width=Pt(1.3)
 if round:sh.adjustments[0]=.10
 if dash:sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
 return sh

def path(pts,c=GRAY,dash=False,arrow=True,width=1.5):
 for i,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
  sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(a[0]),Inches(a[1]),Inches(b[0]),Inches(b[1]));sh.line.color.rgb=RGBColor.from_string(c);sh.line.width=Pt(width)
  if dash:sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
  if arrow and i==len(pts)-2:
   e=OxmlElement('a:tailEnd');e.set('type','triangle');sh._element.spPr.get_or_add_ln().append(e)

def free(pts,x,y,sc,c,width=1.6,close=True):
 fb=s.shapes.build_freeform(pts[0][0],pts[0][1],scale=Inches(sc));fb.add_line_segments(pts[1:],close=close);sh=fb.convert_to_shape(Inches(x),Inches(y));sh.fill.background();sh.line.color.rgb=RGBColor.from_string(c);sh.line.width=Pt(width)
 return sh

def drawlayout(side,x,y,w,h):
 d=json.loads((D/f'{side}_prediction.json').read_text());p=[(-a,b) for a,b in d['single_room_layout']['layout_polygon_pre_manhattan_local_bi_xz']]
 ep=json.loads((D/'pair_prediction_only.json').read_text())[f'portal_geometry_{side}']['endpoints_local']
 allp=p+ep;lo=[min(a[k] for a in allp) for k in [0,1]];hi=[max(a[k] for a in allp) for k in [0,1]];sc=min(w/(hi[0]-lo[0]),h/(hi[1]-lo[1]));xx=x+(w-(hi[0]-lo[0])*sc)/2;yy=y+(h-(hi[1]-lo[1])*sc)/2
 pts=[((a-lo[0])*1000,(hi[1]-b)*1000) for a,b in p];free(pts,xx,yy,sc/1000,BLUE if side=='A' else GREEN)
 pts=[(xx+(a-lo[0])*sc,yy+(hi[1]-b)*sc) for a,b in ep];path(pts,RED,arrow=False,width=3)

text(.5,.19,7,.2,'总体技术路线  |  双房间研究单元',10,GRAY)
text(.5,.52,12.3,.55,'基于全景图的跨房间布局匹配与联合拼接',26,RED,True)
text(.52,1.12,11.9,.3,'视觉建立开口对应，几何完成相对对齐；进一步研究候选条件下的视觉验证。',14,GRAY)
# Top stage labels
for x,w,t,c in [(.5,2.15,'双图输入',GRAY),(3,3.7,'单房间感知',BLUE),(7.25,2.25,'跨图匹配',ORANGE),(10.3,2.55,'联合布局',GREEN)]:text(x,1.60,w,.35,t,18,c,True,True)
# Input + networks + native prediction diagrams.
for side,num,y in [('A',34,2.12),('B',37,4.02)]:
 s.shapes.add_picture(str(D/f'{side}_pano_{num}.jpg'),Inches(.5),Inches(y),width=Inches(2.15),height=Inches(1.075))
 text(.5,y+1.13,2.15,.28,f'全景 {side} · pano {num}',12,GRAY,False,True)
 box(3,y,1.6,1.10)
 # stacked feature maps are editable symbols, not Matcher internals
 for i in range(3):box(3.19+i*.095,y+.18-i*.035,.37,.38,'C6DBFA',BLUE,False,False)
 text(3.66,y+.15,.78,.34,'Eθ',18,BLUE,True,True)
 text(3.08,y+.67,1.44,.3,'Bi-Layout',16,BLUE,True,True)
 path([(2.65,y+.53),(3,y+.53)])
 box(5.04,y-.10,1.66,1.32,'FFFFFF','BACBDF')
 drawlayout(side,5.23,y+.01,1.28,1.05)
 text(4.86,y+1.32,2.02,.45,f'布局 L{side} · 开口 O{side}',13,DARK,False,True)
 path([(4.6,y+.53),(5.04,y+.53)])
 text(4.63,y+.06,.36,.28,'F',12,BLUE,False,True)
text(3,3.43,1.6,.3,'共享权重 θ',13,BLUE,True,True)
path([(3.8,3.23),(3.8,3.4)],BLUE,arrow=False,width=1)
path([(3.8,3.77),(3.8,4.02)],BLUE,arrow=False,width=1)
# Matcher is deliberately opaque.
box(7.25,2.75,2.25,1.28,'FFF1E1','D5A16A')
text(7.36,2.94,2.03,.37,'Cross-Matcher',19,ORANGE,True,True)
text(7.36,3.48,2.03,.4,'跨图匹配、评分与拒绝',13,DARK,False,True)
path([(6.7,2.65),(6.94,2.65),(6.94,3.10),(7.25,3.10)])
path([(6.7,4.55),(6.94,4.55),(6.94,3.69),(7.25,3.69)])
text(6.98,2.19,2.7,.27,'两侧视觉特征 + 开口候选',11,GRAY,False,True)
# Predicted geometry to solver and current direct output.
box(7.25,4.65,2.25,1.02,'F1F2F4','7C858F')
text(7.35,4.80,2.05,.3,'候选相对对齐',17,DARK,True,True)
text(7.35,5.22,2.05,.3,'旋转 R · 平移 t · 尺度 s',12,GRAY,False,True)
path([(8.38,4.03),(8.38,4.65)],width=2)
text(8.48,4.20,1.25,.28,'匹配开口对',12,GRAY)
path([(5.90,5.70),(5.90,5.94),(7.02,5.94),(7.02,5.18),(7.25,5.18)])
text(5.22,5.99,1.77,.25,'预测布局与开口几何',10,GRAY)
# Actual output from this pair, native polygons from frozen SVG.
box(10.3,2.08,2.55,3.60,'F3FAF5','85B697')
sr=ET.parse(D/'joint_layout_original.svg').getroot()
for poly in sr.findall('{http://www.w3.org/2000/svg}polygon'):
 pts=[tuple(map(float,t.split(','))) for t in poly.attrib['points'].split()];free(pts,10.43,2.27,2.28/900,poly.attrib['stroke'][1:],1.6)
ln=sr.find('{http://www.w3.org/2000/svg}line');path([(10.43+float(ln.attrib['x1'])*2.28/900,2.27+float(ln.attrib['y1'])*2.28/900),(10.43+float(ln.attrib['x2'])*2.28/900,2.27+float(ln.attrib['y2'])*2.28/900)],RED,arrow=False,width=3)
text(10.44,4.56,2.27,.3,'统一坐标下的双房间布局',13,GREEN,True,True)
# Small editable topology, same real two-room relation.
box(10.58,5.04,.54,.34,'FFFFFF',BLUE);text(10.62,5.065,.46,.2,'A',12,BLUE,True,True)
box(12.04,5.04,.54,.34,'FFFFFF',GREEN);text(12.08,5.065,.46,.2,'B',12,GREEN,True,True)
path([(11.12,5.21),(12.04,5.21)],RED,arrow=False,width=2);text(11.2,4.92,.76,.22,'共享接口',10,RED,False,True)
path([(9.5,5.14),(9.90,5.14),(9.90,3.58),(10.3,3.58)],width=2)
text(9.62,3.94,.5,.75,'当\n前\n流\n程',10,GRAY,False,True)
# Refusal is existing; use solid gray, reserve dashed lines for future.
box(10.3,6.08,2.55,.61,'F1F2F4','8B9098')
text(10.39,6.19,2.37,.32,'拒绝：保留独立布局',15,GRAY,True,True)
path([(9.5,3.15),(9.75,3.15),(9.75,6.40),(10.3,6.40)],GRAY,width=1)
text(8.93,6.04,1.23,.28,'无可靠对应',11,GRAY,False,True)
# Proposed branch, SALVe attribution in notes and caption. Visual evidence explicitly supplied.
box(.52,6.06,4.05,.63,'E8F0FE',BLUE,True)
text(.65,6.19,3.8,.32,'视觉验证（拟研究）：对齐纹理 / 视觉特征',14,BLUE,True,True)
path([(7.25,5.52),(6.86,5.52),(6.86,6.41),(4.57,6.41)],BLUE,True)
text(5.15,6.45,1.4,.25,'对齐假设',10,BLUE,False,True)
path([(3.20,5.12),(3.20,6.06)],BLUE,True)
text(1.52,5.63,1.57,.28,'两图视觉证据',11,BLUE,False,True)
# planned result goes around bottom, with separate labels rather than crossing solver.
path([(4.57,6.20),(4.83,6.20),(4.83,6.88),(12.99,6.88),(12.99,4.24),(12.85,4.24)],BLUE,True,width=1.15)
text(6.18,6.89,4.3,.23,'验证通过 → 拼接；未通过 → 拒绝',10,BLUE,False,True)
path([(12.99,6.41),(12.85,6.41)],BLUE,True,width=1.15)
text(.52,7.18,10.85,.20,'实线：已有流程   虚线：拟研究   |   图像与布局：T3 / ZInD train / 0665，GT仅用于事后核验；多房间为后续扩展',8.6,GRAY)
text(11.58,7.16,1.2,.2,'可编辑流程图',9,GRAY,False,True)
# Correct notes master body if necessary.
ns=s.notes_slide
if ns.notes_text_frame is None:
 sp=OxmlElement('p:sp');nv=OxmlElement('p:nvSpPr');cn=OxmlElement('p:cNvPr');cn.set('id','99');cn.set('name','Notes');nv.append(cn);nv.append(OxmlElement('p:cNvSpPr'));np=OxmlElement('p:nvPr');ph=OxmlElement('p:ph');ph.set('type','body');ph.set('idx','1');np.append(ph);nv.append(np);sp.append(nv);sp.append(OxmlElement('p:spPr'));tb=OxmlElement('p:txBody');tb.append(OxmlElement('a:bodyPr'));tb.append(OxmlElement('a:lstStyle'));tb.append(OxmlElement('a:p'));sp.append(tb);ns.shapes._spTree.append(sp)
ns.notes_text_frame.text='''本页使用用户上一轮已确认的全景0665/floor03/pano34、37；本轮两个ChatGPT临时图片链接无法读取，不能确认其是否与本图相同。输入JPG原样嵌入。单图及联合布局轮廓从已有预测JSON/SVG逐点等比例绘制为可编辑多边形，未用GT替换预测，未生成新的模型推理。单图显示比例独立，不表达相对米制尺度。该案例属于历史T3、官方train定性诊断，非最新ObservablePairs泛化结果；正确匹配不等于无误差几何。
实线包括已有匹配拒绝路径。虚线只表示拟研究：候选相对对齐产生假设，视觉特征提供纹理或图像特征证据，验证通过后拼接，否则拒绝。图中未画训练损失、特征维度或Matcher内部层。Bi-Layout的叠层矩形为共享视觉编码器的学术示意，不表示精确层数。
代码对应：输入dataset/panorama_pair_dataset.py；双布局models/bi_layout.py::Bi_Layout.forward；开口models/cross_scene_matcher.py::OpeningSignalHead.forward及opening_probabilities_to_intervals；匹配models/cross_scene_matcher.py::OpeningGuidedCrossAttentionMatcher.forward；候选端点utils/candidate_local_portal_geometry.py::estimate_candidate_local_portal_geometry；相对变换与拼接utils/joint_layout.py::build_joint_layout_from_endpoints。开口候选不是Bi-Layout原生输出，而是开口头根据图像特征与双深度生成，图中单房间感知阶段将这些操作合并表示。
创新仍为两项：可靠候选与视觉主导可靠连接；几何求解是辅助模块，不额外宣称第三项创新。新增纹理视觉验证尚未实现，借鉴SALVe（ECCV2022）第4.3节：https://johnwlambert.github.io/salve/ 。只有占据/轮廓栅格不能称为纹理验证；实现时须处理无效投影与未观测区域。图中拒绝是证据不足或不匹配的模型输出，不能一概等同真实不相邻。'''
prs.core_properties.title='跨房间布局匹配与联合拼接总体框架';prs.core_properties.author='孙润坤'
prs.save(O/'跨房间匹配与拼接_单页流程图.pptx')
(O/'说明.md').write_text('一页可编辑PPT。照片为上一轮已核对的pano34与37；本轮两个临时链接无法读取，尚不能确认是同一组图片。除两张全景照片外，图框、文字、连线、特征示意、单图轮廓及联合布局均为原生PowerPoint对象。\n\n'+ns.notes_text_frame.text)
print(O/'跨房间匹配与拼接_单页流程图.pptx')

# Complete three-page presentation. The original first-page file remains available.
# Remove the narrow vertical label on page 1, which could collide with adjacent routes.
for sh in list(prs.slides[0].shapes):
 if sh.has_text_frame and sh.text=='当\n前\n流\n程':prs.slides[0].shapes._spTree.remove(sh._element)

def newpage(title,subtitle,num):
 global s
 s=prs.slides.add_slide(layout)
 for sh in list(s.shapes):s.shapes._spTree.remove(sh._element)
 text(.55,.20,8,.25,f'方法细化  |  {num:02d}',10,GRAY)
 text(.55,.55,12.2,.55,title,26,RED,True)
 text(.57,1.16,12.1,.40,subtitle,14,GRAY)
 text(.55,7.16,11.9,.24,'依据现有代码与冻结输出绘制；T3 / ZInD train / 0665 的案例仅作定性解释；GT不参与预测。',9,GRAY)
 text(12.28,7.12,.45,.30,f'{num:02d}',11,GRAY,False,True)
 return s

def putnotes(t):
 ns=s.notes_slide
 if ns.notes_text_frame is None:
  sp=OxmlElement('p:sp');nv=OxmlElement('p:nvSpPr');cn=OxmlElement('p:cNvPr');cn.set('id','99');cn.set('name','Notes');nv.append(cn);nv.append(OxmlElement('p:cNvSpPr'));np=OxmlElement('p:nvPr');ph=OxmlElement('p:ph');ph.set('type','body');ph.set('idx','1');np.append(ph);nv.append(np);sp.append(nv);sp.append(OxmlElement('p:spPr'));tb=OxmlElement('p:txBody');tb.append(OxmlElement('a:bodyPr'));tb.append(OxmlElement('a:lstStyle'));tb.append(OxmlElement('a:p'));sp.append(tb);ns.shapes._spTree.append(sp)
 ns.notes_text_frame.text=t

# Page 2: opening representation and pair construction, not a second matcher.
newpage('开口匹配细化：融合表示、候选生成与配对','先为两张图分别提出可靠开口，再组织跨图候选集合；对应关系由同一个 Matcher 判断。',2)
steps=[(.65,2.75,'双布局与视觉特征','视觉特征 F\n闭合深度 Denc\n外延深度 Dext','E8F0FE',BLUE),
       (3.78,2.75,'开口融合表示','拼接视觉特征与深度差异\nΔD、相对差异、对数比例','E8F0FE',BLUE),
       (6.91,2.75,'Opening Head','点卷积 → 环形卷积\n学习响应 + 几何先验','FFF1E1',ORANGE),
       (10.04,2.65,'概率与候选区间','开口概率 Popen\n阈值 → 环形连通区间','EAF5EE',GREEN)]
for x,w,title,body,fill,c in steps:
 box(x,1.99,w,1.85,fill,c)
 text(x+.12,2.17,w-.24,.4,title,18,c,True,True)
 text(x+.18,2.78,w-.36,1.00,body,14,DARK,False,True)
for x in [3.40,6.53,9.66]:path([(x,2.90),(x+.38,2.90)])
text(.8,4.18,5.9,.40,'真实预测区间：支持跨全景接缝',19,BLUE,True)
# Each strip is a true native candidate interval. A's three candidates share one strip.
for side,y in [('A',4.89),('B',5.69)]:
 d=json.loads((D/f'{side}_prediction.json').read_text());intervals=d['opening_prediction']['intervals'];x=1.28;w=5.30
 text(.8,y+.04,.35,.3,side,18,DARK,True)
 box(x,y,w,.28,'F1F2F4','D5DAE0',False,False)
 for idx,(a,b) in enumerate(intervals):
  color=RED if (side=='A' and idx==1) or side=='B' else BLUE
  spans=[(a,b+1)] if a<=b else [(a,256),(0,b+1)]
  for start,end in spans:box(x+w*start/256,y,w*(end-start)/256,.28,color,color,False,False)
 text(x,y+.34,w,.27,'  '.join(f'{side}{i}: [{a}, {b}]' for i,(a,b) in enumerate(intervals)),12,GRAY)
text(1.28,6.50,5.3,.34,'横轴为环形特征单元索引；红色是本例最终选中候选。',11,GRAY)
box(7.12,4.27,5.57,2.24,'FFFFFF','B8C6D4')
text(7.32,4.47,5.17,.42,'跨图配对集合：O_A × O_B',21,ORANGE,True)
text(7.32,5.06,5.14,.44,'本例：A 有 3 个候选，B 有 1 个候选',17,DARK)
text(7.32,5.61,5.14,.63,'保留每个候选的区间、有效掩码与视觉特征；\n交给 Cross-Matcher 评分，避免凭几何直接定对应。',15,GRAY)
putnotes('本页细化开口匹配的前端准备，不增加第二个匹配网络。OpeningSignalHead.forward将LayerNorm(F)与delta=relu(Dext-Denc)、relative_delta、log_ratio拼接，经过1x1卷积与circular padding卷积，输出learned_opening_logits和depth_scale_logits；将几何prior logits加入学习logits后sigmoid生成开口概率。现有opening_probabilities_to_intervals支持环形阈值连通区间。图为功能表示，未展示层数/特征维度。候选A:[[240,27],[207,211],[119,122]]，B:[[223,23]]，来自deliverables/correct_match_0665_34_37/A_prediction.json和B_prediction.json，对应原始T3冻结输出。候选条按256个特征单元等比例绘制，不是像素级分割，不使用GT区间。候选集合与视觉特征送入同一Matcher；先token cross-attention后候选池化见下一页。已有环形处理和双布局差异不得作为新原创；未来边界细化与窄开口增强属于拟研究。代码：models/cross_scene_matcher.py::OpeningSignalHead、opening_probabilities_to_intervals；models/bi_layout.py::Bi_Layout.forward。')

# Page 3: implementation-faithful ordering, attention then pooling, then two-way decision.
newpage('Matcher 细化：双向视觉交互、候选评分与拒绝','先在图像特征单元间交互，再在候选区间内池化；不把“最高分”直接等同于“接受匹配”。',3)
for side,y in [('A',2.05),('B',3.69)]:
 box(.68,y,2.25,1.12,'E8F0FE',BLUE)
 text(.80,y+.17,2.01,.35,f'图 {side}：视觉特征 F{side}',17,BLUE,True,True)
 text(.80,y+.67,2.01,.30,'开口权重 + 候选掩码',13,GRAY,False,True)
box(3.58,2.04,3.05,2.78,'E8F0FE',BLUE)
text(3.76,2.22,2.69,.40,'双向交叉注意力',20,BLUE,True,True)
text(3.76,2.91,2.69,.43,'Q_A → K_B, V_B',19,DARK,False,True)
text(3.76,3.48,2.69,.43,'Q_B → K_A, V_A',19,DARK,False,True)
text(3.76,4.10,2.69,.55,'开口引导偏置 + 有效区域\n残差与前馈更新特征',13,GRAY,False,True)
path([(2.93,2.61),(3.58,2.61)]);path([(2.93,4.25),(3.58,4.25)])
box(7.28,2.82,2.12,1.34,'FFF1E1',ORANGE)
text(7.4,3.00,1.88,.40,'候选加权池化',18,ORANGE,True,True)
text(7.4,3.58,1.88,.40,'聚合交互后的特征',13,GRAY,False,True)
path([(6.63,3.49),(7.28,3.49)])
box(10.06,2.10,2.62,2.72,'EAF5EE',GREEN)
text(10.20,2.31,2.34,.42,'候选评分与分配',19,GREEN,True,True)
text(10.25,2.98,2.24,.45,'相似度 → 双向分配',16,DARK,False,True)
text(10.25,3.57,2.24,.43,'加入“无匹配”类别',16,DARK,False,True)
text(10.25,4.20,2.24,.29,'开口置信度辅助候选评分',12,GRAY,False,True)
path([(9.40,3.49),(10.06,3.49)])
# Frozen score bars: real candidate pair scores, not invented attention maps.
text(.8,5.18,5.8,.36,'本例真实候选评分（非校准概率）',18,ORANGE,True)
pp=json.loads((D/'pair_prediction_only.json').read_text())
for i,y in enumerate([5.78,6.17,6.56]):
 val=pp['candidate_pair_scores'][i][0];text(.83,y,1.40,.26,f'A{i} ↔ B0',13,DARK)
 box(2.31,y+.025,2.75,.18,'F1F2F4','F1F2F4',False,False)
 box(2.31,y+.025,max(.015,2.75*val),.18,GREEN if i==1 else BLUE,GREEN if i==1 else BLUE,False,False)
 text(5.23,y,1.28,.27,f'{val:.4f}',13,GREEN if i==1 else GRAY,i==1)
box(7.28,5.28,5.4,1.49,'FFFFFF','879A88')
text(7.48,5.44,5.0,.36,'双向都优于“无匹配”项，才允许接受',17,GREEN,True)
text(7.48,5.99,5.0,.58,'接受集合内选最高分 → 候选几何对齐\n接受集合为空 → 拒绝，不强制拼接',15,DARK)
putnotes('代码顺序已经核对：OpeningGuidedCrossAttentionMatcher._attend计算QK转置的scaled logits，叠加opening_bias_strength*log(P_open)及key candidate mask，softmax后乘V；双向更新得到cross_feature_A/B。随后_candidate_matches调用OpeningTokenPooler，按候选mask和开口权重池化交互后的特征。归一化的候选query/key点积除以temperature形成logits；添加learned dustbin，分别执行A到B和B到A的row softmax。mutual_affinity=sqrt(P_AB(i,j)*P_BA(j,i))，开口置信度由两侧候选内平均Popen组合，形成candidate_pair_score。accepted_pair要求两侧分配均大于对应dustbin，最终在被接受的集合中选最高分。右下框为现有实现，不是新增研究模块。真实候选分数[0.0032921230886131525,0.8816207647323608,0.005463342182338238]来自0665_floor_03_pano_34_pano_37_3.json，去除了padding列，只展示有效A0/A1/A2与B0。该例实际接受A1-B0。match和no-match汇总分数不是互补的已校准二分类概率，不能另设显示分数0.5阈值冒充当前决策。token shift不是相机yaw，位姿在外部解析几何阶段得到。来源：models/cross_scene_matcher.py::OpeningGuidedCrossAttentionMatcher._attend、_candidate_matches、forward；OpeningTokenPooler。')

assert len(prs.slides)==3
out=O/'跨房间匹配与拼接_总体与细化_3页.pptx'
prs.save(out)
# Resave the cleaned first-page standalone copy, preserving its notes.
solo=Presentation(out)
for sid in list(solo.slides._sldIdLst)[1:]:solo.part.drop_rel(sid.rId);solo.slides._sldIdLst.remove(sid)
solo.save(O/'跨房间匹配与拼接_单页流程图.pptx')
(O/'代码对应表.md').write_text('''| 图中模块 | 代码 | 输入 → 输出 |
|---|---|---|
| 双图输入 | dataset/panorama_pair_dataset.py | 图像对 → RGB与样本元数据 |
| 共享单图感知 | models/bi_layout.py::Bi_Layout.forward | RGB → 视觉特征、双深度 |
| 开口融合与候选 | models/cross_scene_matcher.py::OpeningSignalHead、opening_probabilities_to_intervals | 特征+双深度 → 开口概率、环形区间 |
| 跨图交互 | OpeningGuidedCrossAttentionMatcher._attend | Q/K/V+开口引导 → 交互特征 |
| 候选池化 | OpeningTokenPooler | 交互特征+候选mask → 候选表示 |
| 候选评分与拒绝 | OpeningGuidedCrossAttentionMatcher._candidate_matches | 候选表示 → 双向分配、接受或拒绝 |
| 候选相对对齐 | utils/candidate_local_portal_geometry.py::estimate_candidate_local_portal_geometry | 预测区间与深度 → 开口端点 |
| 联合布局 | utils/joint_layout.py::build_joint_layout_from_endpoints | 端点及布局 → B→A变换、联合轮廓 |
| 纹理视觉验证 | 尚未实现；第1页虚线方案 | 候选对齐假设+视觉证据 → 可信度与重排/拒绝 |

第2页与第3页是同一个匹配系统的前端和Matcher细化，没有重复添加网络。原始图像取上一轮已核对的34/37；本轮临时图片链接不可读取。\n''')
problems=[];details=[]
for i,slide in enumerate(prs.slides):
 for sh in slide.shapes:
  if sh.left<0 or sh.top<0 or sh.left+sh.width>prs.slide_width+1000 or sh.top+sh.height>prs.slide_height+1000:problems.append([i+1,sh.name])
 details.append({'page':i+1,'shapes':len(slide.shapes),'pictures':sum(sh.shape_type==13 for sh in slide.shapes),'notes_present':bool(slide.notes_slide.notes_text_frame.text)})
assert not problems,problems
(O/'结构检查.json').write_text(json.dumps({'pages':3,'page_details':details,'out_of_bounds':problems,'rendered':False,'render_blocker':'Automatic approval review rejected LibreOffice rendering because the account usage limit was reached.','font':'Microsoft YaHei; not installed locally; font-faithful rendering not checked','input_images':'Previous verified pair 0665 / floor03 / pano34,37; newly pasted temporary links inaccessible'},ensure_ascii=False,indent=2))
(O/'三页讲解备注.md').write_text('\n\n'.join(f'## 第{i+1}页\n\n'+sl.notes_slide.notes_text_frame.text for i,sl in enumerate(prs.slides)))
print(out)
